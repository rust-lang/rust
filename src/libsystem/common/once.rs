use sync as sys;
use thread;
use core::sync::atomic::{AtomicIsize, Ordering};

pub struct Once {
    cnt: AtomicIsize,
}

impl Once {
    pub const fn new() -> Once { Once { cnt: AtomicIsize::new(0) } }
}

impl sys::Once for Once {
    fn call_once<F: FnOnce()>(&'static self, f: F) {
        if self.cnt.load(Ordering::Relaxed) < 0 {
            return
        }

        match self.cnt.fetch_add(1, Ordering::Acquire) {
            c if c < 0 => (),
            c if c > 0 => while self.cnt.load(Ordering::Relaxed) >= 0 { <thread::imp::Thread as thread::Thread>::yield_() },
            _ => {
                f();
                self.cnt.store(isize::min_value(), Ordering::Relaxed);
            },
        }
    }
}
