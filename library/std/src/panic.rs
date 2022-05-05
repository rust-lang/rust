//! Panic support in the standard library.

#![stable(feature = "std_panic", since = "1.9.0")]

use crate::any::Any;
use crate::collections;
use crate::panicking;
use crate::sync::atomic::{AtomicUsize, Ordering};
use crate::sync::{Mutex, RwLock};
use crate::thread::Result;

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
#[allow_internal_unstable(libstd_sys_internals, const_format_args, core_panic)]
#[cfg_attr(not(test), rustc_diagnostic_item = "std_panic_2015_macro")]
#[rustc_macro_transparency = "semitransparent"]
pub macro panic_2015 {
    () => ({
        $crate::rt::begin_panic("explicit panic")
    }),
    ($msg:expr $(,)?) => ({
        $crate::rt::begin_panic($msg)
    }),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => ({
        $crate::rt::panic_display(&$arg)
    }),
    ($fmt:expr, $($arg:tt)+) => ({
        $crate::rt::panic_fmt($crate::const_format_args!($fmt, $($arg)+))
    }),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
pub use core::panic::panic_2021;

#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use crate::panicking::{set_hook, take_hook};

#[unstable(feature = "panic_update_hook", issue = "92649")]
pub use crate::panicking::update_hook;

#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use core::panic::{Location, PanicInfo};

#[stable(feature = "catch_unwind", since = "1.9.0")]
pub use core::panic::{AssertUnwindSafe, RefUnwindSafe, UnwindSafe};

/// Panic the current thread with the given message as the panic payload.
///
/// The message can be of any (`Any + Send`) type, not just strings.
///
/// The message is wrapped in a `Box<'static + Any + Send>`, which can be
/// accessed later using [`PanicInfo::payload`].
///
/// See the [`panic!`] macro for more information about panicking.
#[stable(feature = "panic_any", since = "1.51.0")]
#[inline]
#[track_caller]
pub fn panic_any<M: 'static + Any + Send>(msg: M) -> ! {
    crate::panicking::begin_panic(msg);
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: ?Sized> UnwindSafe for Mutex<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: ?Sized> UnwindSafe for RwLock<T> {}

#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl<T: ?Sized> RefUnwindSafe for Mutex<T> {}
#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl<T: ?Sized> RefUnwindSafe for RwLock<T> {}

// https://github.com/rust-lang/rust/issues/62301
#[stable(feature = "hashbrown", since = "1.36.0")]
impl<K, V, S> UnwindSafe for collections::HashMap<K, V, S>
where
    K: UnwindSafe,
    V: UnwindSafe,
    S: UnwindSafe,
{
}

/// Invokes a closure, capturing the cause of an unwinding panic if one occurs.
///
/// This function will return `Ok` with the closure's result if the closure
/// does not panic, and will return `Err(cause)` if the closure panics. The
/// `cause` returned is the object with which panic was originally invoked.
///
/// It is currently undefined behavior to unwind from Rust code into foreign
/// code, so this function is particularly useful when Rust is called from
/// another language (normally C). This can run arbitrary Rust code, capturing a
/// panic and allowing a graceful handling of the error.
///
/// It is **not** recommended to use this function for a general try/catch
/// mechanism. The [`Result`] type is more appropriate to use for functions that
/// can fail on a regular basis. Additionally, this function is not guaranteed
/// to catch all panics, see the "Notes" section below.
///
/// The closure provided is required to adhere to the [`UnwindSafe`] trait to ensure
/// that all captured variables are safe to cross this boundary. The purpose of
/// this bound is to encode the concept of [exception safety][rfc] in the type
/// system. Most usage of this function should not need to worry about this
/// bound as programs are naturally unwind safe without `unsafe` code. If it
/// becomes a problem the [`AssertUnwindSafe`] wrapper struct can be used to quickly
/// assert that the usage here is indeed unwind safe.
///
/// [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1236-stabilize-catch-panic.md
///
/// # Notes
///
/// Note that this function **might not catch all panics** in Rust. A panic in
/// Rust is not always implemented via unwinding, but can be implemented by
/// aborting the process as well. This function *only* catches unwinding panics,
/// not those that abort the process.
///
/// Also note that unwinding into Rust code with a foreign exception (e.g.
/// an exception thrown from C++ code) is undefined behavior.
///
/// # Examples
///
/// ```
/// use std::panic;
///
/// let result = panic::catch_unwind(|| {
///     println!("hello!");
/// });
/// assert!(result.is_ok());
///
/// let result = panic::catch_unwind(|| {
///     panic!("oh no!");
/// });
/// assert!(result.is_err());
/// ```
#[stable(feature = "catch_unwind", since = "1.9.0")]
pub fn catch_unwind<F: FnOnce() -> R + UnwindSafe, R>(f: F) -> Result<R> {
    unsafe { panicking::r#try(f) }
}

/// Triggers a panic without invoking the panic hook.
///
/// This is designed to be used in conjunction with [`catch_unwind`] to, for
/// example, carry a panic across a layer of C code.
///
/// # Notes
///
/// Note that panics in Rust are not always implemented via unwinding, but they
/// may be implemented by aborting the process. If this function is called when
/// panics are implemented this way then this function will abort the process,
/// not trigger an unwind.
///
/// # Examples
///
/// ```should_panic
/// use std::panic;
///
/// let result = panic::catch_unwind(|| {
///     panic!("oh no!");
/// });
///
/// if let Err(err) = result {
///     panic::resume_unwind(err);
/// }
/// ```
#[stable(feature = "resume_unwind", since = "1.9.0")]
pub fn resume_unwind(payload: Box<dyn Any + Send>) -> ! {
    panicking::rust_panic_without_hook(payload)
}

/// Make all future panics abort directly without running the panic hook or unwinding.
///
/// There is no way to undo this; the effect lasts until the process exits or
/// execs (or the equivalent).
///
/// # Use after fork
///
/// This function is particularly useful for calling after `libc::fork`.  After `fork`, in a
/// multithreaded program it is (on many platforms) not safe to call the allocator.  It is also
/// generally highly undesirable for an unwind to unwind past the `fork`, because that results in
/// the unwind propagating to code that was only ever expecting to run in the parent.
///
/// `panic::always_abort()` helps avoid both of these.  It directly avoids any further unwinding,
/// and if there is a panic, the abort will occur without allocating provided that the arguments to
/// panic can be formatted without allocating.
///
/// Examples
///
/// ```no_run
/// #![feature(panic_always_abort)]
/// use std::panic;
///
/// panic::always_abort();
///
/// let _ = panic::catch_unwind(|| {
///     panic!("inside the catch");
/// });
///
/// // We will have aborted already, due to the panic.
/// unreachable!();
/// ```
#[unstable(feature = "panic_always_abort", issue = "84438")]
pub fn always_abort() {
    crate::panicking::panic_count::set_always_abort();
}

/// The configuration for whether and how the default panic hook will capture
/// and display the backtrace.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[unstable(feature = "panic_backtrace_config", issue = "93346")]
#[non_exhaustive]
pub enum BacktraceStyle {
    /// Prints a terser backtrace which ideally only contains relevant
    /// information.
    Short,
    /// Prints a backtrace with all possible information.
    Full,
    /// Disable collecting and displaying backtraces.
    Off,
}

impl BacktraceStyle {
    pub(crate) fn full() -> Option<Self> {
        if cfg!(feature = "backtrace") { Some(BacktraceStyle::Full) } else { None }
    }

    fn as_usize(self) -> usize {
        match self {
            BacktraceStyle::Short => 1,
            BacktraceStyle::Full => 2,
            BacktraceStyle::Off => 3,
        }
    }

    fn from_usize(s: usize) -> Option<Self> {
        Some(match s {
            0 => return None,
            1 => BacktraceStyle::Short,
            2 => BacktraceStyle::Full,
            3 => BacktraceStyle::Off,
            _ => unreachable!(),
        })
    }
}

// Tracks whether we should/can capture a backtrace, and how we should display
// that backtrace.
//
// Internally stores equivalent of an Option<BacktraceStyle>.
static SHOULD_CAPTURE: AtomicUsize = AtomicUsize::new(0);

/// Configure whether the default panic hook will capture and display a
/// backtrace.
///
/// The default value for this setting may be set by the `RUST_BACKTRACE`
/// environment variable; see the details in [`get_backtrace_style`].
#[unstable(feature = "panic_backtrace_config", issue = "93346")]
pub fn set_backtrace_style(style: BacktraceStyle) {
    if !cfg!(feature = "backtrace") {
        // If the `backtrace` feature of this crate isn't enabled, skip setting.
        return;
    }
    SHOULD_CAPTURE.store(style.as_usize(), Ordering::Release);
}

/// Checks whether the standard library's panic hook will capture and print a
/// backtrace.
///
/// This function will, if a backtrace style has not been set via
/// [`set_backtrace_style`], read the environment variable `RUST_BACKTRACE` to
/// determine a default value for the backtrace formatting:
///
/// The first call to `get_backtrace_style` may read the `RUST_BACKTRACE`
/// environment variable if `set_backtrace_style` has not been called to
/// override the default value. After a call to `set_backtrace_style` or
/// `get_backtrace_style`, any changes to `RUST_BACKTRACE` will have no effect.
///
/// `RUST_BACKTRACE` is read according to these rules:
///
/// * `0` for `BacktraceStyle::Off`
/// * `full` for `BacktraceStyle::Full`
/// * `1` for `BacktraceStyle::Short`
/// * Other values are currently `BacktraceStyle::Short`, but this may change in
///   the future
///
/// Returns `None` if backtraces aren't currently supported.
#[unstable(feature = "panic_backtrace_config", issue = "93346")]
pub fn get_backtrace_style() -> Option<BacktraceStyle> {
    if !cfg!(feature = "backtrace") {
        // If the `backtrace` feature of this crate isn't enabled quickly return
        // `Unsupported` so this can be constant propagated all over the place
        // to optimize away callers.
        return None;
    }
    if let Some(style) = BacktraceStyle::from_usize(SHOULD_CAPTURE.load(Ordering::Acquire)) {
        return Some(style);
    }

    // Setting environment variables for Fuchsia components isn't a standard
    // or easily supported workflow. For now, display backtraces by default.
    let format = if cfg!(target_os = "fuchsia") {
        BacktraceStyle::Full
    } else {
        crate::env::var_os("RUST_BACKTRACE")
            .map(|x| {
                if &x == "0" {
                    BacktraceStyle::Off
                } else if &x == "full" {
                    BacktraceStyle::Full
                } else {
                    BacktraceStyle::Short
                }
            })
            .unwrap_or(BacktraceStyle::Off)
    };
    set_backtrace_style(format);
    Some(format)
}

#[cfg(test)]
mod tests;
