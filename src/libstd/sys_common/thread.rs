use boxed::FnBox;
use env;
use sync::atomic::{self, Ordering};
use sys::stack_overflow;
use sys::thread as imp;

#[allow(dead_code)]
pub unsafe fn start_thread(main: *mut u8) {
    // Next, set up our stack overflow handler which may get triggered if we run
    // out of stack.
    let _handler = stack_overflow::Handler::new();

    // Finally, let's run some code.
    Box::from_raw(main as *mut Box<dyn FnBox()>)()
}

pub fn min_stack() -> usize {
    static MIN: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
    match MIN.load(Ordering::SeqCst) {
        0 => {}
        n => return n - 1,
    }
    let amt = env::var("RUST_MIN_STACK").ok().and_then(|s| s.parse().ok());
    let amt = amt.unwrap_or(imp::DEFAULT_MIN_STACK_SIZE);

    // 0 is our sentinel value, so ensure that we'll never see 0 after
    // initialization has run
    MIN.store(amt + 1, Ordering::SeqCst);
    amt
}
