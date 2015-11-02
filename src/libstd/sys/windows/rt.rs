use sys::windows::stack_overflow;

pub use sys::common::rt::{std_cleanup, min_stack};
pub use libc::strlen;

pub unsafe fn run_main<R, F: FnOnce() -> R>(f: F, _argc: isize, _argv: *const *const u8) -> R {
    stack_overflow::init();

    let ret = f();

    cleanup();

    ret
}

pub unsafe fn run_thread<R, F: FnOnce() -> R>(f: F) -> R {
    let _handler = stack_overflow::Handler::new();

    let ret = f();
    thread_cleanup();
    ret
}

pub unsafe fn cleanup() {
    use sync::Once;

    static CLEANUP: Once = Once::new();
    CLEANUP.call_once(|| {
        stack_overflow::cleanup();
    });
}

pub unsafe fn thread_cleanup() {
}
