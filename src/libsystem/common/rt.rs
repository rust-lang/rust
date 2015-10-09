use libc;
use core::fmt;

pub type c_char = libc::c_char;
pub use libc::strlen;

pub fn min_stack() -> usize {
    use env::prelude::*;
    use os_str::prelude::*;
    use core::sync::atomic;

    static MIN: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
    match MIN.load(atomic::Ordering::Relaxed) {
        0 => {}
        n => return n - 1,
    }
    let amt = Env::getenv(OsStr::from_str("RUST_MIN_STACK")).unwrap_or(None).and_then(|s| s.into_string().ok()).and_then(|s| s.parse().ok());
    let amt = amt.unwrap_or(2 * 1024 * 1024);
    // 0 is our sentinel value, so ensure that we'll never see 0 after
    // initialization has run
    MIN.store(amt + 1, atomic::Ordering::Relaxed);
    amt
}

pub fn abort(args: fmt::Arguments) -> ! {
    use core::intrinsics;
    use stdio;

    stdio::dumb_print(format_args!("fatal runtime error: {}", args));
    unsafe { intrinsics::abort(); }
}

#[macro_export]
macro_rules! rtabort {
    ($($t:tt)*) => (<$crate::rt::prelude::Runtime as $crate::rt::Runtime>::abort(format_args!($($t)*)))
}

#[macro_export]
macro_rules! rtassert {
    ($e:expr) => ({
        if !$e {
            rtabort!(concat!("assertion failed: ", stringify!($e)))
        }
    })
}
