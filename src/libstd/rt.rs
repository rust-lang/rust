//! Runtime services
//!
//! The `rt` module provides a narrow set of runtime services,
//! including the global heap (exported in `heap`) and unwinding and
//! backtrace support. The APIs in this module are highly unstable,
//! and should be considered as private implementation details for the
//! time being.

#![unstable(feature = "rt",
            reason = "this public module should not exist and is highly likely \
                      to disappear",
            issue = "0")]
#![doc(hidden)]


// Re-export some of our utilities which are expected by other crates.
pub use crate::panicking::{begin_panic, begin_panic_fmt, update_panic_count};

// To reduce the generated code of the new `lang_start`, this function is doing
// the real work.
#[cfg(not(test))]
fn lang_start_internal(main: &(dyn Fn() -> i32 + Sync + crate::panic::RefUnwindSafe),
                       argc: isize, argv: *const *const u8) -> isize {
    use crate::panic;
    use crate::sys;
    use crate::sys_common;
    use crate::sys_common::thread_info;
    use crate::thread::Thread;

    sys::init();

    unsafe {
        let main_guard = sys::thread::guard::init();
        sys::stack_overflow::init();

        // Next, set up the current Thread with the guard information we just
        // created. Note that this isn't necessary in general for new threads,
        // but we just do this to name the main thread and to give it correct
        // info about the stack bounds.
        let thread = Thread::new(Some("main".to_owned()));
        thread_info::set(main_guard, thread);

        // Store our args if necessary in a squirreled away location
        sys::args::init(argc, argv);

        // Let's run some code!
        #[cfg(feature = "backtrace")]
        let exit_code = panic::catch_unwind(|| {
            sys_common::backtrace::__rust_begin_short_backtrace(move || main())
        });
        #[cfg(not(feature = "backtrace"))]
        let exit_code = panic::catch_unwind(move || main());

        sys_common::cleanup();
        exit_code.unwrap_or(101) as isize
    }
}

#[cfg(not(test))]
#[lang = "start"]
fn lang_start<T: crate::process::Termination + 'static>
    (main: fn() -> T, argc: isize, argv: *const *const u8) -> isize
{
    lang_start_internal(&move || main().report(), argc, argv)
}
