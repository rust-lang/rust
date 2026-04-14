use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

pub static CONSOLE: Mutex<Option<FbConsole>> = Mutex::new(None);
pub static CONSOLE_DISABLED: AtomicBool = AtomicBool::new(false);

pub struct FbConsole;

unsafe impl Send for FbConsole {}

impl FbConsole {
    pub fn new() -> Self {
        Self
    }

    pub fn put_char(&mut self, _c: u8) {}
}

pub fn init() {
    *CONSOLE.lock() = Some(FbConsole::new());
}

pub fn disable() {
    CONSOLE_DISABLED.store(true, Ordering::Relaxed);
}

pub fn is_disabled() -> bool {
    CONSOLE_DISABLED.load(Ordering::Relaxed)
}

pub fn put_char(c: u8) {
    if CONSOLE_DISABLED.load(Ordering::Relaxed) {
        return;
    }
    if let Some(ref mut console) = *CONSOLE.lock() {
        console.put_char(c);
    }
}
