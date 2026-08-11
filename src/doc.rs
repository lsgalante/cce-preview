//! Document model + background page rasterization for cce-preview.
//!
//! A document is a list of pages with known sizes: PDFs report points via
//! `pdfinfo` and rasterize per page through `pdftoppm` (poppler), raster
//! images are single-page documents sized in pixels. Workers decode or
//! rasterize off-thread, upload RGBA via `cce_ui::vk::upload_rgba` (the
//! upload queue is thread-safe), then notify the app over the calloop
//! channel so the engine wakes and repaints.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{mpsc, Arc, Mutex};

use crate::Message;

/// Extensions the `image` crate is built to decode (keep in sync with the
/// feature list in Cargo.toml).
pub const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "ico"];

/// GPU pages kept resident. The cce-ui image registry hard-caps at 256
/// images total, so leave generous headroom.
const MAX_GPU_PAGES: usize = 24;
/// Largest bitmap edge we'll upload; bigger sources are downscaled (images)
/// or rendered at a capped DPI (PDF pages).
const MAX_DIM: u32 = 8192;
const RENDER_THREADS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Image,
    Pdf,
}

/// Page size in document units: points for PDFs, pixels for images.
#[derive(Debug, Clone, Copy)]
pub struct PageSize {
    pub w: f64,
    pub h: f64,
}

pub struct Document {
    pub path: PathBuf,
    pub kind: Kind,
    pub pages: Vec<PageSize>,
}

impl Document {
    pub fn load(path: &Path) -> Result<Self, String> {
        let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
        match ext.as_deref() {
            Some("pdf") => Self::load_pdf(path),
            Some(e) if IMAGE_EXTS.contains(&e) => Self::load_image(path),
            _ => Err("unsupported file type".to_string()),
        }
    }

    fn load_image(path: &Path) -> Result<Self, String> {
        let (w, h) = image::image_dimensions(path).map_err(|e| e.to_string())?;
        Ok(Self {
            path: path.to_path_buf(),
            kind: Kind::Image,
            pages: vec![PageSize { w: w as f64, h: h as f64 }],
        })
    }

    /// Page count from `pdfinfo`, then per-page sizes from a second ranged
    /// call. `pdfinfo` reports MediaBox dimensions with a separate `rot`
    /// field, while `pdftoppm` bakes /Rotate into its output — so swap
    /// width/height here for 90°/270° pages to keep layout and pixels agreed.
    fn load_pdf(path: &Path) -> Result<Self, String> {
        let count_out = pdfinfo(path, &[])?;
        let count: usize = count_out
            .lines()
            .find_map(|l| l.strip_prefix("Pages:"))
            .and_then(|v| v.trim().parse().ok())
            .ok_or("pdfinfo: no page count")?;
        if count == 0 {
            return Err("empty PDF".to_string());
        }
        let sizes_out = pdfinfo(path, &["-f", "1", "-l", &count.to_string()])?;
        let mut sizes: Vec<PageSize> = Vec::with_capacity(count);
        let mut rots: Vec<i32> = Vec::with_capacity(count);
        for line in sizes_out.lines() {
            let Some(rest) = line.strip_prefix("Page ") else { continue };
            let Some((_, field)) = rest.trim_start().split_once(' ') else { continue };
            if let Some(v) = field.trim_start().strip_prefix("size:") {
                // "595.276 x 841.89 pts (A4)"
                let mut it = v.trim().split_whitespace();
                let w: f64 = it.next().and_then(|s| s.parse().ok()).ok_or("pdfinfo: bad size")?;
                let h: f64 = it.nth(1).and_then(|s| s.parse().ok()).ok_or("pdfinfo: bad size")?;
                sizes.push(PageSize { w, h });
            } else if let Some(v) = field.trim_start().strip_prefix("rot:") {
                rots.push(v.trim().parse().unwrap_or(0));
            }
        }
        if sizes.len() != count {
            return Err(format!("pdfinfo: {} sizes for {count} pages", sizes.len()));
        }
        for (s, rot) in sizes.iter_mut().zip(rots) {
            if rot == 90 || rot == 270 {
                std::mem::swap(&mut s.w, &mut s.h);
            }
        }
        Ok(Self { path: path.to_path_buf(), kind: Kind::Pdf, pages: sizes })
    }
}

fn pdfinfo(path: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("pdfinfo")
        .args(args)
        .arg(path)
        .output()
        .map_err(|e| format!("pdfinfo: {e}"))?;
    if !out.status.success() {
        return Err(format!("pdfinfo: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// A rendered page as delivered by a worker.
#[derive(Debug, Clone, Copy)]
pub struct Rendered {
    pub image: u32,
    pub dpi: u32,
}

struct Job {
    generation: u64,
    page: usize,
    dpi: u32,
    path: PathBuf,
    kind: Kind,
    size: PageSize,
    /// Extra user rotation in quarter turns cw, applied to the pixels.
    quarter_turns: u8,
}

enum PageState {
    Pending,
    Ready { r: Rendered, refreshing: bool, last_used: u64 },
    Failed,
}

/// Per-document GPU page cache: lazy render requests, DPI upgrades, LRU
/// eviction. `reset()` bumps the generation so late results from a previous
/// document/rotation are freed on arrival instead of displayed.
pub struct PageStore {
    states: HashMap<usize, PageState>,
    queue: mpsc::Sender<Job>,
    generation: u64,
    frame: u64,
}

impl PageStore {
    pub fn new(notify: calloop::channel::Sender<Message>) -> Self {
        let (queue, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..RENDER_THREADS {
            let rx = Arc::clone(&rx);
            let notify = notify.clone();
            std::thread::spawn(move || worker(rx, notify));
        }
        Self { states: HashMap::new(), queue, generation: 0, frame: 0 }
    }

    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    pub fn reset(&mut self) {
        for (_, state) in self.states.drain() {
            if let PageState::Ready { r, .. } = state {
                cce_ui::vk::free_image(r.image);
            }
        }
        self.generation += 1;
    }

    /// The page's GPU image if resident (marks it used, queues a DPI upgrade
    /// when the resident render is stale); otherwise queues a render (once)
    /// and returns None.
    pub fn ensure(&mut self, doc: &Document, quarter_turns: u8, page: usize, want_dpi: u32) -> Option<Rendered> {
        let job = |dpi| Job {
            generation: self.generation,
            page,
            dpi,
            path: doc.path.clone(),
            kind: doc.kind,
            size: doc.pages[page],
            quarter_turns,
        };
        match self.states.get_mut(&page) {
            Some(PageState::Ready { r, refreshing, last_used }) => {
                *last_used = self.frame;
                if r.dpi != want_dpi && doc.kind == Kind::Pdf && !*refreshing {
                    *refreshing = true;
                    let _ = self.queue.send(job(want_dpi));
                }
                Some(*r)
            }
            Some(_) => None,
            None => {
                self.states.insert(page, PageState::Pending);
                let _ = self.queue.send(job(want_dpi));
                None
            }
        }
    }

    pub fn complete(&mut self, generation: u64, page: usize, result: Option<Rendered>) {
        if generation != self.generation {
            if let Some(r) = result {
                cce_ui::vk::free_image(r.image);
            }
            return;
        }
        let state = match result {
            Some(r) => PageState::Ready { r, refreshing: false, last_used: self.frame },
            None => PageState::Failed,
        };
        if let Some(PageState::Ready { r, .. }) = self.states.insert(page, state) {
            cce_ui::vk::free_image(r.image);
        }
        self.evict();
    }

    /// Free the least-recently-used pages once over budget; pages touched
    /// this frame are never evicted.
    fn evict(&mut self) {
        let resident = self.states.values().filter(|s| matches!(s, PageState::Ready { .. })).count();
        if resident <= MAX_GPU_PAGES {
            return;
        }
        let mut ready: Vec<(usize, u64)> = self
            .states
            .iter()
            .filter_map(|(p, s)| match s {
                PageState::Ready { last_used, .. } if *last_used < self.frame => Some((*p, *last_used)),
                _ => None,
            })
            .collect();
        ready.sort_by_key(|&(_, used)| used);
        for (page, _) in ready.into_iter().take(resident - MAX_GPU_PAGES) {
            if let Some(PageState::Ready { r, .. }) = self.states.remove(&page) {
                cce_ui::vk::free_image(r.image);
            }
        }
    }
}

fn worker(rx: Arc<Mutex<mpsc::Receiver<Job>>>, notify: calloop::channel::Sender<Message>) {
    loop {
        let job = match rx.lock().unwrap().recv() {
            Ok(j) => j,
            Err(_) => return,
        };
        let result = render(&job)
            .map_err(|e| log::warn!("{}: page {}: {e}", job.path.display(), job.page + 1))
            .ok();
        let msg = Message::Page { generation: job.generation, page: job.page, result };
        if notify.send(msg).is_err() {
            return;
        }
    }
}

fn render(job: &Job) -> Result<Rendered, String> {
    let mut rgba = match job.kind {
        Kind::Image => {
            let img = image::open(&job.path).map_err(|e| e.to_string())?;
            let mut rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            if w.max(h) > MAX_DIM {
                let s = MAX_DIM as f64 / w.max(h) as f64;
                let (nw, nh) = (((w as f64 * s) as u32).max(1), ((h as f64 * s) as u32).max(1));
                rgba = image::imageops::resize(&rgba, nw, nh, image::imageops::FilterType::Triangle);
            }
            rgba
        }
        Kind::Pdf => {
            // Cap the DPI so the page bitmap stays under MAX_DIM on its
            // longer edge (page size is in points, 72/inch).
            let max_pts = job.size.w.max(job.size.h).max(1.0);
            let dpi = (job.dpi as f64).min(MAX_DIM as f64 * 72.0 / max_pts).max(18.0) as u32;
            let page = (job.page + 1).to_string();
            // No output root: poppler's pdftoppm writes the PNG to stdout.
            let out = Command::new("pdftoppm")
                .args(["-png", "-r", &dpi.to_string(), "-f", &page, "-l", &page])
                .arg(&job.path)
                .output()
                .map_err(|e| format!("pdftoppm: {e}"))?;
            if !out.status.success() || out.stdout.is_empty() {
                return Err(format!("pdftoppm: {}", String::from_utf8_lossy(&out.stderr).trim()));
            }
            image::load_from_memory(&out.stdout).map_err(|e| e.to_string())?.to_rgba8()
        }
    };
    match job.quarter_turns % 4 {
        1 => rgba = image::imageops::rotate90(&rgba),
        2 => rgba = image::imageops::rotate180(&rgba),
        3 => rgba = image::imageops::rotate270(&rgba),
        _ => {}
    }
    let (w, h) = rgba.dimensions();
    let image = cce_ui::vk::upload_rgba(rgba.into_raw(), w, h);
    Ok(Rendered { image, dpi: job.dpi })
}
