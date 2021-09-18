//! Runtime services
//!
//! The `rt` module provides a narrow set of runtime services,
//! including the global heap (exported in `heap`) and unwinding and
//! backtrace support. The APIs in this module are highly unstable,
//! and should be considered as private implementation details for the
//! time being.

#![unstable(
    feature = "rt",
    reason = "this public module should not exist and is highly likely \
              to disappear",
    issue = "none"
)]
#![doc(hidden)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(unused_macros)]

use crate::ffi::CString;

// Re-export some of our utilities which are expected by other crates.
pub use crate::panicking::{begin_panic, begin_panic_fmt, panic_count};
pub use core::panicking::panic_display;

use crate::sync::Once;
use crate::sys;
use crate::sys_common::thread_info;
use crate::thread::Thread;

// Prints to the "panic output", depending on the platform this may be:
// - the standard error output
// - some dedicated platform specific output
// - nothing (so this macro is a no-op)
macro_rules! rtprintpanic {
    ($($t:tt)*) => {
        if let Some(mut out) = crate::sys::stdio::panic_output() {
            let _ = crate::io::Write::write_fmt(&mut out, format_args!($($t)*));
        }
    }
}

macro_rules! rtabort {
    ($($t:tt)*) => {
        {
            rtprintpanic!("fatal runtime error: {}\n", format_args!($($t)*));
            crate::sys::abort_internal();
        }
    }
}

macro_rules! rtassert {
    ($e:expr) => {
        if !$e {
            rtabort!(concat!("assertion failed: ", stringify!($e)));
        }
    };
}

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

// One-time runtime initialization.
// Runs before `main`.
// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
#[cfg_attr(test, allow(dead_code))]
unsafe fn init(argc: isize, argv: *const *const u8) {
    unsafe {
        sys::init(argc, argv);

        let main_guard = sys::thread::guard::init();
        // Next, set up the current Thread with the guard information we just
        // created. Note that this isn't necessary in general for new threads,
        // but we just do this to name the main thread and to give it correct
        // info about the stack bounds.
        let thread = Thread::new(Some(rtunwrap!(Ok, CString::new("main"))));
        thread_info::set(main_guard, thread);
    }
}

// One-time runtime cleanup.
// Runs after `main` or at program exit.
// NOTE: this is not guaranteed to run, for example when the program aborts.
pub(crate) fn cleanup() {
    static CLEANUP: Once = Once::new();
    CLEANUP.call_once(|| unsafe {
        // Flush stdout and disable buffering.
        crate::io::cleanup();
        // SAFETY: Only called once during runtime cleanup.
        sys::cleanup();
    });
}

// To reduce the generated code of the new `lang_start`, this function is doing
// the real work.
#[cfg(not(test))]
fn lang_start_internal(
    main: &(dyn Fn() -> i32 + Sync + crate::panic::RefUnwindSafe),
    argc: isize,
    argv: *const *const u8,
) -> Result<isize, !> {
    use crate::{mem, panic};
    let rt_abort = move |e| {
        mem::forget(e);
        rtabort!("initialization or cleanup bug");
    };
    // Guard against the code called by this function from unwinding outside of the Rust-controlled
    // code, which is UB. This is a requirement imposed by a combination of how the
    // `#[lang="start"]` attribute is implemented as well as by the implementation of the panicking
    // mechanism itself.
    //
    // There are a couple of instances where unwinding can begin. First is inside of the
    // `rt::init`, `rt::cleanup` and similar functions controlled by libstd. In those instances a
    // panic is a libstd implementation bug. A quite likely one too, as there isn't any way to
    // prevent libstd from accidentally introducing a panic to these functions. Another is from
    // user code from `main` or, more nefariously, as described in e.g. issue #86030.
    // SAFETY: Only called once during runtime initialization.
    panic::catch_unwind(move || unsafe { init(argc, argv) }).map_err(rt_abort)?;
    let ret_code = panic::catch_unwind(move || panic::catch_unwind(main).unwrap_or(101) as isize)
        .map_err(move |e| {
            mem::forget(e);
            rtabort!("drop of the panic payload panicked");
        });
    panic::catch_unwind(cleanup).map_err(rt_abort)?;
    ret_code
}

#[cfg(not(test))]
#[lang = "start"]
fn lang_start<T: crate::process::Termination + 'static>(
    main: fn() -> T,
    argc: isize,
    argv: *const *const u8,
) -> isize {
    let Ok(v) = lang_start_internal(
        &move || crate::sys_common::backtrace::__rust_begin_short_backtrace(main).report(),
        argc,
        argv,
    );
    v
}
