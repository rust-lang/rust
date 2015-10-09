use thread::prelude::*;

use rt as sys;
use common;
use unix;
use libc;
use libc::funcs::posix01::signal::signal;
use core::any::Any;
use core::fmt;

pub use common::rt::{c_char, strlen};

pub struct Runtime(());

impl sys::Runtime for Runtime {
    unsafe fn run_main<R, F: FnOnce() -> R>(f: F, argc: isize, argv: *const *const u8) -> R {
        // By default, some platforms will send a *signal* when a EPIPE error
        // would otherwise be delivered. This runtime doesn't install a SIGPIPE
        // handler, causing it to kill the program, which isn't exactly what we
        // want!
        //
        // Hence, we set SIGPIPE to ignore when the program starts up in order
        // to prevent this problem.
        assert!(signal(libc::SIGPIPE, libc::SIG_IGN) == 0);

        Thread::guard_init();

        common::args::init(argc, argv);

        let ret = f();

        Self::cleanup();

        ret
    }

    unsafe fn run_thread<R, F: FnOnce() -> R>(f: F) -> R {
        Thread::guard_current();
        let ret = f();
        Self::thread_cleanup();
        ret
    }

    unsafe fn cleanup() {
        use sync::prelude::*;

        static CLEANUP: Once = Once::new();
        CLEANUP.call_once(|| unsafe {
            common::args::cleanup();
            unix::stack_overflow::cleanup();
        });
    }

    unsafe fn thread_cleanup() {
    }

    #[inline(always)]
    fn on_panic(msg: &(Any + Send), file: &'static str, line: u32) {
        extern {
            fn rust_std_on_panic(msg: &Any, file: &'static str, line: u32);
        }

        unsafe { rust_std_on_panic(msg, file, line) }
    }

    fn min_stack() -> usize {
        common::rt::min_stack()
    }

    fn abort(args: fmt::Arguments) -> ! {
        common::rt::abort(args)
    }
}
