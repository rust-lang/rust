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

#[rustfmt::skip]
pub use crate::panicking::{begin_panic, panic_count};
pub use core::panicking::{panic_display, panic_fmt};

#[rustfmt::skip]
use crate::any::Any;
use crate::sync::Once;
use crate::thread::{self, Thread};
use crate::{mem, panic, sys};

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

fn handle_rt_panic<T>(e: Box<dyn Any + Send>) -> T {
    mem::forget(e);
    rtabort!("initialization or cleanup bug");
}

// One-time runtime initialization.
// Runs before `main`.
// SAFETY: must be called only once during runtime initialization.
// NOTE: this is not guaranteed to run, for example when Rust code is called externally.
//
// # The `sigpipe` parameter
//
// Since 2014, the Rust runtime on Unix has set the `SIGPIPE` handler to
// `SIG_IGN`. Applications have good reasons to want a different behavior
// though, so there is a `-Zon-broken-pipe` compiler flag that
// can be used to select how `SIGPIPE` shall be setup (if changed at all) before
// `fn main()` is called. See <https://github.com/rust-lang/rust/issues/97889>
// for more info.
//
// The `sigpipe` parameter to this function gets its value via the code that
// rustc generates to invoke `fn lang_start()`. The reason we have `sigpipe` for
// all platforms and not only Unix, is because std is not allowed to have `cfg`
// directives as this high level. See the module docs in
// `src/tools/tidy/src/pal.rs` for more info. On all other platforms, `sigpipe`
// has a value, but its value is ignored.
//
// Even though it is an `u8`, it only ever has 4 values. These are documented in
// `compiler/rustc_session/src/config/sigpipe.rs`.
#[cfg_attr(test, allow(dead_code))]
unsafe fn init(argc: isize, argv: *const *const u8, sigpipe: u8) {
    #[cfg_attr(target_os = "teeos", allow(unused_unsafe))]
    unsafe {
        sys::init(argc, argv, sigpipe)
    };

    // Set up the current thread handle to give it the right name.
    //
    // When code running before main uses `ReentrantLock` (for example by
    // using `println!`), the thread ID can become initialized before we
    // create this handle. Since `set_current` fails when the ID of the
    // handle does not match the current ID, we should attempt to use the
    // current thread ID here instead of unconditionally creating a new
    // one. Also see #130210.
    let thread = unsafe { Thread::new_main(thread::current_id()) };
    if let Err(_thread) = thread::set_current(thread) {
        // `thread::current` will create a new handle if none has been set yet.
        // Thus, if someone uses it before main, this call will fail. That's a
        // bad idea though, as we then cannot set the main thread name here.
        //
        // FIXME: detect the main thread in `thread::current` and use the
        //        correct name there.
        rtabort!("code running before main must not use thread::current");
    }
}

/// Clean up the thread-local runtime state. This *should* be run after all other
/// code managed by the Rust runtime, but will not cause UB if that condition is
/// not fulfilled. Also note that this function is not guaranteed to be run, but
/// skipping it will cause leaks and therefore is to be avoided.
pub(crate) fn thread_cleanup() {
    // This function is run in situations where unwinding leads to an abort
    // (think `extern "C"` functions). Abort here instead so that we can
    // print a nice message.
    panic::catch_unwind(|| {
        crate::thread::drop_current();
    })
    .unwrap_or_else(handle_rt_panic);
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
    sigpipe: u8,
) -> isize {
    // Guard against the code called by this function from unwinding outside of the Rust-controlled
    // code, which is UB. This is a requirement imposed by a combination of how the
    // `#[lang="start"]` attribute is implemented as well as by the implementation of the panicking
    // mechanism itself.
    //
    // There are a couple of instances where unwinding can begin. First is inside of the
    // `rt::init`, `rt::cleanup` and similar functions controlled by bstd. In those instances a
    // panic is a std implementation bug. A quite likely one too, as there isn't any way to
    // prevent std from accidentally introducing a panic to these functions. Another is from
    // user code from `main` or, more nefariously, as described in e.g. issue #86030.
    //
    // We use `catch_unwind` with `handle_rt_panic` instead of `abort_unwind` to make the error in
    // case of a panic a bit nicer.
    panic::catch_unwind(move || {
        // SAFETY: Only called once during runtime initialization.
        unsafe { init(argc, argv, sigpipe) };

        let ret_code = panic::catch_unwind(main).unwrap_or_else(move |payload| {
            // Carefully dispose of the panic payload.
            let payload = panic::AssertUnwindSafe(payload);
            panic::catch_unwind(move || drop({ payload }.0)).unwrap_or_else(move |e| {
                mem::forget(e); // do *not* drop the 2nd payload
                rtabort!("drop of the panic payload panicked");
            });
            // Return error code for panicking programs.
            101
        });
        let ret_code = ret_code as isize;

        cleanup();
        // Guard against multiple threads calling `libc::exit` concurrently.
        // See the documentation for `unique_thread_exit` for more information.
        crate::sys::exit_guard::unique_thread_exit();

        ret_code
    })
    .unwrap_or_else(handle_rt_panic)
}

#[cfg(not(any(test, doctest)))]
#[lang = "start"]
fn lang_start<T: crate::process::Termination + 'static>(
    main: fn() -> T,
    argc: isize,
    argv: *const *const u8,
    sigpipe: u8,
) -> isize {
    lang_start_internal(
        &move || crate::sys::backtrace::__rust_begin_short_backtrace(main).report().to_i32(),
        argc,
        argv,
        sigpipe,
    )
}
