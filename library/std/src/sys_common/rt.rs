#![deny(unsafe_op_in_unsafe_fn)]

use crate::sync::Once;
use crate::sys;
use crate::sys_common::thread_info;
use crate::thread::Thread;

// One-time runtime initialization.
// Runs before `main`.
// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
#[cfg_attr(test, allow(dead_code))]
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    unsafe {
        sys::init(argc, argv);

        let main_guard = sys::thread::guard::init();
        // Next, set up the current Thread with the guard information we just
        // created. Note that this isn't necessary in general for new threads,
        // but we just do this to name the main thread and to give it correct
        // info about the stack bounds.
        let thread = Thread::new(Some("main".to_owned()));
        thread_info::set(main_guard, thread);
    }
}

// One-time runtime cleanup.
// Runs after `main` or at program exit.
// NOTE: this is not guaranteed to run, for example when the program aborts.
#[cfg_attr(test, allow(dead_code))]
pub fn cleanup() {
    static CLEANUP: Once = Once::new();
    CLEANUP.call_once(|| unsafe {
        // Flush stdout and disable buffering.
        crate::io::cleanup();
        // SAFETY: Only called once during runtime cleanup.
        sys::cleanup();
    });
}

macro_rules! rtabort {
    ($($t:tt)*) => (crate::sys_common::util::abort(format_args!($($t)*)))
}

macro_rules! rtassert {
    ($e:expr) => {
        if !$e {
            rtabort!(concat!("assertion failed: ", stringify!($e)));
        }
    };
}

#[allow(unused_macros)] // not used on all platforms
macro_rules! rtunwrap {
    ($ok:ident, $e:expr) => {
        match $e {
            $ok(v) => v,
            ref err => {
                let err = err.as_ref().map(drop); // map Ok/Some which might not be Debug
                rtabort!(concat!("unwrap failed: ", stringify!($e), " = {:?}"), err)
            }
        }
    };
}
