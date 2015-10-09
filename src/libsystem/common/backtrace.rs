use env::prelude::*;
use os_str::prelude::*;
use core::sync::atomic;

// For now logging is turned off by default, and this function checks to see
// whether the magical environment variable is present to see if it's turned on.
pub fn log_enabled() -> bool {
    static ENABLED: atomic::AtomicIsize = atomic::AtomicIsize::new(0);
    match ENABLED.load(atomic::Ordering::SeqCst) {
        1 => return false,
        2 => return true,
        _ => {}
    }

    let val = match Env::getenv(OsStr::from_str("RUST_BACKTRACE")) {
        Ok(Some(..)) => 2,
        _ => 1,
    };
    ENABLED.store(val, atomic::Ordering::SeqCst);
    val == 2
}
