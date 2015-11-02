use sys::unix::{args, stack_overflow};
use sys::thread::Thread;
use libc;

pub use libc::strlen;
pub use sys::common::rt::{std_cleanup, min_stack};

pub unsafe fn run_main<R, F: FnOnce() -> R>(f: F, argc: isize, argv: *const *const u8) -> R {
    #[cfg(not(target_os = "nacl"))]
    unsafe fn ignore_sigpipe() {
        use libc::funcs::posix01::signal::signal;

        // By default, some platforms will send a *signal* when a EPIPE error
        // would otherwise be delivered. This runtime doesn't install a SIGPIPE
        // handler, causing it to kill the program, which isn't exactly what we
        // want!
        //
        // Hence, we set SIGPIPE to ignore when the program starts up in order
        // to prevent this problem.
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) != !0);
    }

    #[cfg(target_os = "nacl")]
    unsafe fn ignore_sigpipe() { }

    stack_overflow::init();

    ignore_sigpipe();

    Thread::guard_init();

    args::init(argc, argv);

    let ret = f();

    cleanup();

    ret
}

pub unsafe fn run_thread<R, F: FnOnce() -> R>(f: F) -> R {
    let _handler = stack_overflow::Handler::new();

    Thread::guard_current();
    let ret = f();
    thread_cleanup();
    ret
}

pub unsafe fn cleanup() {
    use sync::Once;

    static CLEANUP: Once = Once::new();
    CLEANUP.call_once(|| {
        args::cleanup();
        stack_overflow::cleanup();
    });
}

pub unsafe fn thread_cleanup() {
}
