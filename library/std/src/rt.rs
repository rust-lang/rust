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

// Re-export some of our utilities which are expected by other crates.
pub use crate::panicking::{begin_panic, begin_panic_fmt, panic_count};

// To reduce the generated code of the new `lang_start`, this function is doing
// the real work.
#[cfg(not(test))]
fn lang_start_internal(
    main: &(dyn Fn() -> i32 + Sync + crate::panic::RefUnwindSafe),
    argc: isize,
    argv: *const *const u8,
) -> Result<isize, !> {
    use crate::{mem, panic, sys, sys_common};
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
    panic::catch_unwind(move || unsafe { sys_common::rt::init(argc, argv) }).map_err(rt_abort)?;
    let ret_code = panic::catch_unwind(move || panic::catch_unwind(main).unwrap_or(101) as isize)
        .map_err(move |e| {
            mem::forget(e);
            rtprintpanic!("drop of the panic payload panicked");
            sys::abort_internal()
        });
    panic::catch_unwind(sys_common::rt::cleanup).map_err(rt_abort)?;
    ret_code
}

#[cfg(not(test))]
#[lang = "start"]
fn lang_start<T: crate::process::Termination + 'static>(
    main: fn() -> T,
    argc: isize,
    argv: *const *const u8,
) -> isize {
    lang_start_internal(
        &move || crate::sys_common::backtrace::__rust_begin_short_backtrace(main).report(),
        argc,
        argv,
    )
    .into_ok()
}
