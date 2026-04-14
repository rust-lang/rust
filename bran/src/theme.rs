use core::sync::atomic::{AtomicBool, Ordering};

/// Legacy boot-display compatibility shim.
pub static THEME_DISABLED: AtomicBool = AtomicBool::new(false);

pub fn init(_fb: crate::framebuffer::Framebuffer) {}

pub fn disable() {
    THEME_DISABLED.store(true, Ordering::Relaxed);
}

pub fn tick(_now_ms: u64) {}

pub fn try_tick(_now_ms: u64) {}

pub fn putchar(_c: u8) {}
