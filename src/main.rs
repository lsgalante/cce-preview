//! cce-preview — document and image viewer in the spirit of macOS Preview.
//!
//! One continuous vertically-scrolled document: PDF pages (rasterized
//! lazily per page via poppler's pdftoppm, re-rendered at higher DPI as
//! you zoom) or a single raster image. View state is a zoom factor
//! (screen px per document unit) plus a scroll offset; pages are laid out
//! in document units so zoom-at-pointer is an exact rescale.
//!
//! Keys: o open · +/- zoom · 0 fit · 1 actual size · r/l rotate ·
//! arrows/PageUp/PageDown pages (or prev/next file for images) · q quit.
//! Wheel scrolls, ctrl+wheel and pinch zoom at the pointer, drag pans.

mod doc;

use std::path::{Path, PathBuf};

use wayland_client::QueueHandle;

use cce_ui::engine::{Application, EngineState, LogicalPosition, LogicalSize, WindowSettings};
use cce_ui::scene::layout::Rect;
use cce_ui::scene::paint::{DisplayList, PaintCtx};
use cce_ui::widget::scroll_motion::{Bounds, ScrollMotion};
use cce_ui::widget::{ElementState, Key, KeyEvent, MouseButton, MouseScrollDelta, NamedKey, Position};

use doc::{Document, PageStore, Rendered, IMAGE_EXTS};

/// Vertical gap between pages, in document units (so the layout scales
/// uniformly with zoom and anchored zooming stays exact).
const GAP_UNITS: f64 = 12.0;
/// Margin left around a fitted page.
const FIT_MARGIN: f64 = 24.0;
const WHEEL_SCROLL_PX: f64 = 48.0;
const KEY_SCROLL_PX: f64 = 80.0;
const ZOOM_MIN: f64 = 0.05;
const ZOOM_MAX: f64 = 16.0;
/// DPI steps pages are rendered at; bucketing keeps small zoom jitters from
/// re-rasterizing every page.
const DPI_BUCKETS: &[u32] = &[36, 48, 72, 96, 144, 192, 288, 384, 576];

#[derive(Debug, Clone)]
enum Message {
    Page { generation: u64, page: usize, result: Option<Rendered> },
    Quit,
}

struct PreviewApp {
    store: PageStore,
    doc: Option<Document>,
    error: Option<String>,
    /// Sibling files for ArrowLeft/Right browsing (CLI args, or the images
    /// in the opened file's directory).
    files: Vec<PathBuf>,
    file_idx: usize,
    /// User rotation in quarter turns clockwise, whole-document.
    quarter_turns: u8,
    /// Screen px per document unit (pt for PDFs, source px for images).
    zoom: f64,
    /// Scroll offset in screen px; 0 when the content fits the window.
    scroll: (f64, f64),
    /// Drives `scroll` (the drawn value) from the wheel: notches glide,
    /// fingers track 1:1 and fling on the lift. Drag, keyboard and zoom
    /// write `scroll` directly; the motion adopts those through `reconcile`.
    scroll_motion: ScrollMotion,
    /// Refit on resize until the user zooms manually.
    fit: bool,
    win: (f32, f32),
    scale: f64,
    pointer: (f64, f64),
    drag: Option<(f64, f64)>,
    ctrl: bool,
    shift: bool,
    /// Whether a renderer has been handed over yet — the first one is the
    /// process's own, any later one is a replacement after a reconnect.
    /// See `renderer_init`.
    seen_renderer: bool,
}

/// Per-page layout rect in document units.
struct PageRect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl PreviewApp {
    fn rotated(&self, page: doc::PageSize) -> (f64, f64) {
        if self.quarter_turns % 2 == 1 {
            (page.h, page.w)
        } else {
            (page.w, page.h)
        }
    }

    /// Page rects stacked vertically, centered in the content width.
    fn layout(&self) -> (Vec<PageRect>, f64, f64) {
        let Some(doc) = &self.doc else { return (Vec::new(), 0.0, 0.0) };
        let content_w = doc.pages.iter().map(|p| self.rotated(*p).0).fold(0.0, f64::max);
        let mut rects = Vec::with_capacity(doc.pages.len());
        let mut y = 0.0;
        for page in &doc.pages {
            let (w, h) = self.rotated(*page);
            rects.push(PageRect { x: (content_w - w) / 2.0, y, w, h });
            y += h + GAP_UNITS;
        }
        (rects, content_w, y - GAP_UNITS)
    }

    /// Top-left of the content in screen coords: centered when it fits,
    /// scrolled when it doesn't.
    fn origin(&self, content_w: f64, content_h: f64) -> (f64, f64) {
        let (w, h) = (self.win.0 as f64, self.win.1 as f64);
        let ox = ((w - content_w * self.zoom) / 2.0).max(0.0) - self.scroll.0;
        let oy = ((h - content_h * self.zoom) / 2.0).max(0.0) - self.scroll.1;
        (ox, oy)
    }

    fn clamp_scroll(&mut self) {
        let (_, cw, ch) = self.layout();
        let (w, h) = (self.win.0 as f64, self.win.1 as f64);
        self.scroll.0 = self.scroll.0.clamp(0.0, (cw * self.zoom - w).max(0.0));
        self.scroll.1 = self.scroll.1.clamp(0.0, (ch * self.zoom - h).max(0.0));
    }

    fn scroll_by(&mut self, dx: f64, dy: f64) {
        self.scroll.0 += dx;
        self.scroll.1 += dy;
        self.clamp_scroll();
    }

    /// The wheel's range per axis, `0..=overflow` — what `clamp_scroll` clamps to.
    fn scroll_bounds(&self) -> (Bounds, Bounds) {
        let (_, cw, ch) = self.layout();
        let (w, h) = (self.win.0 as f64, self.win.1 as f64);
        (Bounds::max((cw * self.zoom - w) as f32), Bounds::max((ch * self.zoom - h) as f32))
    }

    /// Copy the motion's position into `scroll` exactly (the f32 round-trips
    /// losslessly, so the next `reconcile` sees no host write).
    fn sync_scroll_from_motion(&mut self) {
        self.scroll = (self.scroll_motion.x.pos() as f64, self.scroll_motion.y.pos() as f64);
    }

    /// Per-frame wheel glide/coast; true while `scroll` is still moving, so
    /// the frame loop keeps drawing.
    fn tick_scroll(&mut self, dt: f32) -> bool {
        self.scroll_motion.reconcile(self.scroll.0 as f32, self.scroll.1 as f32);
        if !self.scroll_motion.is_animating() {
            return false;
        }
        let (bx, by) = self.scroll_bounds();
        let moved = self.scroll_motion.tick(dt, bx, by);
        self.sync_scroll_from_motion();
        moved || self.scroll_motion.is_animating()
    }

    /// Multiply zoom, keeping the document point under (px, py) fixed.
    fn zoom_at(&mut self, factor: f64, px: f64, py: f64) {
        let (_, cw, ch) = self.layout();
        let (ox, oy) = self.origin(cw, ch);
        let (dx, dy) = ((px - ox) / self.zoom, (py - oy) / self.zoom);
        self.zoom = (self.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        self.fit = false;
        let (w, h) = (self.win.0 as f64, self.win.1 as f64);
        let pad_x = ((w - cw * self.zoom) / 2.0).max(0.0);
        let pad_y = ((h - ch * self.zoom) / 2.0).max(0.0);
        self.scroll.0 = pad_x - (px - dx * self.zoom);
        self.scroll.1 = pad_y - (py - dy * self.zoom);
        self.clamp_scroll();
    }

    /// The page overlapping the viewport center (for HUD and refit).
    fn current_page(&self) -> usize {
        let (rects, cw, ch) = self.layout();
        let (_, oy) = self.origin(cw, ch);
        let mid = (self.win.1 as f64 / 2.0 - oy) / self.zoom;
        rects
            .iter()
            .position(|r| mid < r.y + r.h + GAP_UNITS / 2.0)
            .unwrap_or(rects.len().saturating_sub(1))
    }

    /// Fit the given page inside the window and scroll to its top.
    fn fit_page(&mut self, page: usize) {
        let (rects, _, _) = self.layout();
        let Some(r) = rects.get(page) else { return };
        let (w, h) = ((self.win.0 as f64 - FIT_MARGIN).max(64.0), (self.win.1 as f64 - FIT_MARGIN).max(64.0));
        self.zoom = (w / r.w).min(h / r.h).clamp(ZOOM_MIN, ZOOM_MAX);
        self.fit = true;
        self.scroll = (0.0, r.y * self.zoom);
        self.clamp_scroll();
    }

    fn go_to_page(&mut self, page: usize) {
        let (rects, _, _) = self.layout();
        if let Some(r) = rects.get(page) {
            self.scroll.1 = (r.y - GAP_UNITS / 2.0) * self.zoom;
            self.clamp_scroll();
        }
    }

    fn open(&mut self, path: &Path, rebuild_collection: bool) {
        self.store.reset();
        self.quarter_turns = 0;
        self.error = None;
        match Document::load(path) {
            Ok(d) => {
                self.doc = Some(d);
                self.fit_page(0);
            }
            Err(e) => {
                self.doc = None;
                self.error = Some(format!("{}: {e}", path.display()));
            }
        }
        if rebuild_collection {
            (self.files, self.file_idx) = collection_for(path);
        }
    }

    fn open_sibling(&mut self, step: i64) {
        if self.files.len() < 2 {
            return;
        }
        let n = self.files.len() as i64;
        self.file_idx = ((self.file_idx as i64 + step).rem_euclid(n)) as usize;
        let path = self.files[self.file_idx].clone();
        self.open(&path, false);
    }

    fn open_dialog(&mut self) {
        let filters: &[(&str, &[&str])] = &[
            ("Documents & images", &["pdf", "png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "ico"]),
            ("PDF", &["pdf"]),
            ("Images", IMAGE_EXTS),
        ];
        if let Some(path) = cce_ui::file_dialog::pick_file("Open", filters) {
            self.open(&path, true);
        }
    }

    fn rotate(&mut self, quarter_turns_cw: i8) {
        self.quarter_turns = (self.quarter_turns as i8 + quarter_turns_cw).rem_euclid(4) as u8;
        self.store.reset();
        self.clamp_scroll();
        if self.fit {
            self.fit_page(self.current_page());
        }
    }

    fn notches(delta: &MouseScrollDelta) -> (f64, f64) {
        match delta {
            MouseScrollDelta::LineDelta(x, y) => (*x as f64, *y as f64),
            MouseScrollDelta::PixelDelta(Position { x, y }) => (x / 60.0, y / 60.0),
        }
    }
}

/// The file's siblings for arrow-key browsing: images in the same
/// directory, name-sorted, with the opened file's position. PDFs browse
/// their own pages instead, so they get a singleton collection.
fn collection_for(path: &Path) -> (Vec<PathBuf>, usize) {
    let is_image = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()));
    if !is_image {
        return (vec![path.to_path_buf()], 0);
    }
    let mut files: Vec<PathBuf> = path
        .parent()
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        })
        .collect();
    files.sort();
    let idx = files.iter().position(|p| p == path).unwrap_or(0);
    if files.is_empty() {
        (vec![path.to_path_buf()], 0)
    } else {
        (files, idx)
    }
}

impl Application for PreviewApp {
    type Message = Message;

    fn new(_qh: &QueueHandle<EngineState<Self>>, sender: calloop::channel::Sender<Self::Message>) -> Self {
        let mut app = Self {
            store: PageStore::new(sender),
            doc: None,
            error: None,
            files: Vec::new(),
            file_idx: 0,
            quarter_turns: 0,
            zoom: 1.0,
            scroll: (0.0, 0.0),
            scroll_motion: ScrollMotion::new(),
            fit: true,
            win: (900.0, 700.0),
            scale: 1.0,
            pointer: (0.0, 0.0),
            drag: None,
            ctrl: false,
            shift: false,
            seen_renderer: false,
        };
        let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
        match args.len() {
            0 => {}
            1 => app.open(&args[0].clone(), true),
            _ => {
                app.files = args;
                app.file_idx = 0;
                let path = app.files[0].clone();
                app.open(&path, false);
            }
        }
        app
    }

    fn settings(&self) -> WindowSettings {
        let title = match &self.doc {
            Some(d) => {
                let name = d.path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                if d.pages.len() > 1 {
                    format!("{name} (page {}/{}) — Preview", self.current_page() + 1, d.pages.len())
                } else {
                    format!("{name} — Preview")
                }
            }
            None => "Preview".to_string(),
        };
        WindowSettings {
            title,
            app_id: "cce-preview".to_string(),
            width: 900,
            height: 700,
            fullscreen: false,
            min_size: Some((320, 240)),
        }
    }

    fn update(&mut self, msg: Self::Message, needs_rebuild: &mut bool, exit: &mut bool) {
        match msg {
            Message::Page { generation, page, result } => {
                self.store.complete(generation, page, result);
                *needs_rebuild = true;
            }
            Message::Quit => *exit = true,
        }
    }

    fn tick(&mut self, dt: f32, needs_rebuild: &mut bool) {
        if self.tick_scroll(dt) {
            *needs_rebuild = true;
        }
    }

    /// Throw the resident pages away when the renderer is replaced.
    ///
    /// `PageStore` holds **renderer** image ids, and a renderer does not
    /// outlive its session: `cce-ui`'s `window_runner` repairs a lost Wayland
    /// transport by opening a new session around the same `Application`, which
    /// rebuilds the renderer and with it the image table. The cached ids then
    /// name images that no longer exist, and a draw for an unknown id is
    /// skipped rather than reported — so a reconnected viewer came back with
    /// its chrome and a blank document, and stayed that way, because a
    /// resident page is never re-rendered.
    ///
    /// `reset` is exactly the right hammer: it frees every page (a free for an
    /// id the new renderer never had is a no-op) and bumps the generation, so
    /// a render still in flight for the old session is dropped on arrival
    /// instead of landing as a page nobody asked for. The next `display_list`
    /// finds nothing resident and queues the visible pages again.
    ///
    /// Not on the first renderer: the pages queued from `new()` are waiting
    /// for precisely that one.
    fn renderer_init(&mut self, _renderer: &mut cce_ui::vk::VkRenderer) {
        if std::mem::replace(&mut self.seen_renderer, true) {
            log::info!("[preview] renderer replaced; re-rendering the resident pages");
            self.store.reset();
        }
    }

    fn handle_resize(&mut self, width: f32, height: f32, scale: f64) {
        self.win = (width, height);
        self.scale = scale;
        if self.fit {
            self.fit_page(self.current_page());
        } else {
            self.clamp_scroll();
        }
    }

    fn handle_pointer_move(&mut self, pos: LogicalPosition, needs_rebuild: &mut bool) {
        let (px, py) = (pos.x as f64, pos.y as f64);
        if let Some((lx, ly)) = self.drag {
            self.scroll_by(lx - px, ly - py);
            self.drag = Some((px, py));
            *needs_rebuild = true;
        }
        self.pointer = (px, py);
    }

    fn handle_mouse_input(
        &mut self,
        button: MouseButton,
        state: ElementState,
        pos: LogicalPosition,
        _needs_rebuild: &mut bool,
    ) -> Option<Self::Message> {
        if button == MouseButton::Left {
            self.drag = match state {
                ElementState::Pressed => Some((pos.x as f64, pos.y as f64)),
                ElementState::Released => None,
            };
        }
        None
    }

    fn handle_mouse_wheel(&mut self, delta: &MouseScrollDelta, pos: LogicalPosition, needs_rebuild: &mut bool) {
        if self.ctrl {
            // Zoom stays instant: a notch (or 60px of finger) is one 1.1 step.
            let (_, ny) = Self::notches(delta);
            if ny != 0.0 {
                self.zoom_at(1.1f64.powf(ny), pos.x as f64, pos.y as f64);
                *needs_rebuild = true;
            }
            return;
        }
        // Plain wheel: a 2-D scroll through the motion — a notch is
        // WHEEL_SCROLL_PX, pixel deltas are 1:1; shift turns the vertical
        // motion horizontal. `tick_scroll` carries `scroll` after it.
        let line = WHEEL_SCROLL_PX as f32;
        let (mut dx, mut dy) = ScrollMotion::delta_px(delta, (line, line));
        if self.shift {
            dx = dy;
            dy = 0.0;
        }
        let discrete = matches!(delta, MouseScrollDelta::LineDelta(..));
        let (bx, by) = self.scroll_bounds();
        self.scroll_motion.reconcile(self.scroll.0 as f32, self.scroll.1 as f32);
        if self.scroll_motion.apply_px(dx, dy, discrete, bx, by) {
            self.sync_scroll_from_motion();
            *needs_rebuild = true;
        }
    }

    fn handle_pinch(&mut self, factor: f32, pos: LogicalPosition, needs_rebuild: &mut bool) -> bool {
        if factor > 0.0 && factor != 1.0 {
            self.zoom_at(factor as f64, pos.x as f64, pos.y as f64);
            *needs_rebuild = true;
        }
        true
    }

    fn handle_key_input(&mut self, event: &KeyEvent, needs_rebuild: &mut bool) -> Option<Self::Message> {
        // Wheel events carry no modifiers, so track ctrl/shift from the key
        // stream for ctrl+wheel zoom / shift+wheel horizontal scroll.
        match &event.logical_key {
            Key::Named(NamedKey::Control) => self.ctrl = event.state == ElementState::Pressed,
            Key::Named(NamedKey::Shift) => self.shift = event.state == ElementState::Pressed,
            _ => {
                self.ctrl = event.ctrl;
                self.shift = event.shift;
            }
        }
        if event.state != ElementState::Pressed {
            return None;
        }
        log::debug!("key: {:?} text={:?} ctrl={} shift={}", event.logical_key, event.text, event.ctrl, event.shift);
        let (cx, cy) = (self.win.0 as f64 / 2.0, self.win.1 as f64 / 2.0);
        let pages = self.doc.as_ref().map_or(0, |d| d.pages.len());
        let file_nav = pages <= 1 && self.files.len() > 1;
        let mut handled = true;
        match &event.logical_key {
            Key::Character(c) if c == "+" || c == "=" => self.zoom_at(1.25, cx, cy),
            Key::Character(c) if c == "-" => self.zoom_at(0.8, cx, cy),
            Key::Character(c) if c == "0" => self.fit_page(self.current_page()),
            Key::Character(c) if c == "1" => {
                let f = 1.0 / self.zoom;
                self.zoom_at(f, cx, cy);
            }
            Key::Character(c) if c == "r" || c == "R" => self.rotate(1),
            Key::Character(c) if c == "l" || c == "L" => self.rotate(-1),
            Key::Character(c) if c == "o" => self.open_dialog(),
            Key::Character(c) if c == "q" => return Some(Message::Quit),
            Key::Named(NamedKey::ArrowUp) => self.scroll_by(0.0, -KEY_SCROLL_PX),
            Key::Named(NamedKey::ArrowDown) => self.scroll_by(0.0, KEY_SCROLL_PX),
            Key::Named(NamedKey::ArrowLeft) if file_nav => self.open_sibling(-1),
            Key::Named(NamedKey::ArrowRight) if file_nav => self.open_sibling(1),
            Key::Named(NamedKey::ArrowLeft) | Key::Named(NamedKey::PageUp) => {
                let p = self.current_page();
                self.go_to_page(p.saturating_sub(1));
            }
            Key::Named(NamedKey::ArrowRight) | Key::Named(NamedKey::PageDown) => {
                let p = self.current_page();
                self.go_to_page((p + 1).min(pages.saturating_sub(1)));
            }
            Key::Named(NamedKey::Space) => self.scroll_by(0.0, self.win.1 as f64 * 0.9),
            Key::Named(NamedKey::Home) => self.go_to_page(0),
            Key::Named(NamedKey::End) => self.go_to_page(pages.saturating_sub(1)),
            _ => handled = false,
        }
        if handled {
            *needs_rebuild = true;
        }
        None
    }

    fn display_list(&mut self, size: LogicalSize, scale: f64) -> Option<DisplayList> {
        self.win = (size.width, size.height);
        self.scale = scale;
        self.store.begin_frame();
        let mut pc = PaintCtx::new();
        pc.quad(Rect { x: 0.0, y: 0.0, width: size.width, height: size.height }, self.clear_color());

        if self.doc.is_none() {
            let msg = self.error.as_deref().unwrap_or("Press 'o' to open a file");
            pc.text(msg, 24.0, size.height / 2.0 - 8.0, 14.0, [180, 180, 180]);
            return Some(pc.finish());
        }

        let (rects, cw, ch) = self.layout();
        let (ox, oy) = self.origin(cw, ch);
        let want_dpi = {
            let want = 72.0 * self.zoom * scale;
            *DPI_BUCKETS
                .iter()
                .find(|&&b| want <= b as f64 * 1.01)
                .unwrap_or(DPI_BUCKETS.last().unwrap())
        };

        let mut visible = Vec::new();
        for (i, r) in rects.iter().enumerate() {
            let rect = Rect {
                x: (ox + r.x * self.zoom) as f32,
                y: (oy + r.y * self.zoom) as f32,
                width: (r.w * self.zoom) as f32,
                height: (r.h * self.zoom) as f32,
            };
            if rect.y > size.height || rect.y + rect.height < 0.0 {
                continue;
            }
            visible.push((i, rect));
        }
        let doc = self.doc.take().unwrap();
        for (i, rect) in &visible {
            // White page ground: placeholder while rendering, and backing
            // for images with transparency.
            pc.quad(
                Rect { x: rect.x - 1.0, y: rect.y - 1.0, width: rect.width + 2.0, height: rect.height + 2.0 },
                [0.0, 0.0, 0.0, 0.35],
            );
            pc.quad(*rect, [0.97, 0.97, 0.97, 1.0]);
            if let Some(r) = self.store.ensure(&doc, self.quarter_turns, *i, want_dpi) {
                pc.image(r.image, *rect, 1.0);
            }
        }
        self.doc = Some(doc);

        // HUD: file name, page, zoom (top-left chip).
        let doc = self.doc.as_ref().unwrap();
        let name = doc.path.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let mut hud = name.to_string();
        if doc.pages.len() > 1 {
            hud.push_str(&format!("   ·   page {}/{}", self.current_page() + 1, doc.pages.len()));
        } else if self.files.len() > 1 {
            hud.push_str(&format!("   ·   {}/{}", self.file_idx + 1, self.files.len()));
        }
        hud.push_str(&format!("   ·   {:.0}%", self.zoom * 100.0));
        let w = 24.0 + hud.chars().count() as f32 * 6.6;
        pc.quad(Rect { x: 8.0, y: 8.0, width: w, height: 24.0 }, [0.0, 0.0, 0.0, 0.45]);
        pc.text(hud, 16.0, 13.0, 12.0, [230, 230, 230]);

        Some(pc.finish())
    }

    fn display_list_text(&self) -> bool {
        true
    }

    fn clear_color(&self) -> [f32; 4] {
        [0.13, 0.13, 0.14, 1.0]
    }
}

fn main() {
    env_logger::init();
    cce_ui::engine::run::<PreviewApp>();
}
