pub fn std_cleanup() {
    ::at_exit::cleanup();
}

pub fn min_stack() -> usize {
    use env;
    use sync::atomic::{AtomicUsize, Ordering};

    static MIN: AtomicUsize = AtomicUsize::new(0);
    match MIN.load(Ordering::Relaxed) {
        0 => {}
        n => return n - 1,
    }
    let amt = env::var("RUST_MIN_STACK").ok()
        .and_then(|s| s.parse().ok());
    let amt = amt.unwrap_or(2 * 1024 * 1024);
    // 0 is our sentinel value, so ensure that we'll never see 0 after
    // initialization has run
    MIN.store(amt + 1, Ordering::Relaxed);
    amt
}

#[macro_export]
macro_rules! rtabort {
    ($($t:tt)*) => ($crate::panicking::abort(format_args!($($t)*)))
}

#[macro_export]
macro_rules! rtassert {
    ($e:expr) => ({
        if !$e {
            rtabort!(concat!("assertion failed: ", stringify!($e)))
        }
    })
}
