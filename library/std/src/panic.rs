//! Panic support in the standard library.

#![stable(feature = "std_panic", since = "1.9.0")]

use crate::any::Any;
use crate::sync::atomic::{AtomicU8, Ordering};
use crate::sync::{Condvar, Mutex, RwLock};
use crate::thread::Result;
use crate::{collections, fmt, panicking};

#[stable(feature = "panic_hooks", since = "1.10.0")]
#[deprecated(
    since = "1.82.0",
    note = "use `PanicHookInfo` instead",
    suggestion = "std::panic::PanicHookInfo"
)]
/// A struct providing information about a panic.
///
/// `PanicInfo` has been renamed to [`PanicHookInfo`] to avoid confusion with
/// [`core::panic::PanicInfo`].
pub type PanicInfo<'a> = PanicHookInfo<'a>;

/// A struct providing information about a panic.
///
/// `PanicHookInfo` structure is passed to a panic hook set by the [`set_hook`] function.
///
/// # Examples
///
/// ```should_panic
/// use std::panic;
///
/// panic::set_hook(Box::new(|panic_info| {
///     println!("panic occurred: {panic_info}");
/// }));
///
/// panic!("critical system failure");
/// ```
///
/// [`set_hook`]: ../../std/panic/fn.set_hook.html
#[stable(feature = "panic_hook_info", since = "1.81.0")]
#[derive(Debug)]
pub struct PanicHookInfo<'a> {
    payload: &'a (dyn Any + Send),
    location: &'a Location<'a>,
    can_unwind: bool,
    force_no_backtrace: bool,
}

impl<'a> PanicHookInfo<'a> {
    #[inline]
    pub(crate) fn new(
        location: &'a Location<'a>,
        payload: &'a (dyn Any + Send),
        can_unwind: bool,
        force_no_backtrace: bool,
    ) -> Self {
        PanicHookInfo { payload, location, can_unwind, force_no_backtrace }
    }

    /// Returns the payload associated with the panic.
    ///
    /// This will commonly, but not always, be a `&'static str` or [`String`].
    ///
    /// A invocation of the `panic!()` macro in Rust 2021 or later will always result in a
    /// panic payload of type `&'static str` or `String`.
    ///
    /// Only an invocation of [`panic_any`]
    /// (or, in Rust 2018 and earlier, `panic!(x)` where `x` is something other than a string)
    /// can result in a panic payload other than a `&'static str` or `String`.
    ///
    /// [`String`]: ../../std/string/struct.String.html
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
    ///         println!("panic occurred: {s:?}");
    ///     } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
    ///         println!("panic occurred: {s:?}");
    ///     } else {
    ///         println!("panic occurred");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn payload(&self) -> &(dyn Any + Send) {
        self.payload
    }

    /// Returns the payload associated with the panic, if it is a string.
    ///
    /// This returns the payload if it is of type `&'static str` or `String`.
    ///
    /// A invocation of the `panic!()` macro in Rust 2021 or later will always result in a
    /// panic payload where `payload_as_str` returns `Some`.
    ///
    /// Only an invocation of [`panic_any`]
    /// (or, in Rust 2018 and earlier, `panic!(x)` where `x` is something other than a string)
    /// can result in a panic payload where `payload_as_str` returns `None`.
    ///
    /// # Example
    ///
    /// ```should_panic
    /// #![feature(panic_payload_as_str)]
    ///
    /// std::panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(s) = panic_info.payload_as_str() {
    ///         println!("panic occurred: {s:?}");
    ///     } else {
    ///         println!("panic occurred");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[inline]
    #[unstable(feature = "panic_payload_as_str", issue = "125175")]
    pub fn payload_as_str(&self) -> Option<&str> {
        if let Some(s) = self.payload.downcast_ref::<&str>() {
            Some(s)
        } else if let Some(s) = self.payload.downcast_ref::<String>() {
            Some(s)
        } else {
            None
        }
    }

    /// Returns information about the location from which the panic originated,
    /// if available.
    ///
    /// This method will currently always return [`Some`], but this may change
    /// in future versions.
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use std::panic;
    ///
    /// panic::set_hook(Box::new(|panic_info| {
    ///     if let Some(location) = panic_info.location() {
    ///         println!("panic occurred in file '{}' at line {}",
    ///             location.file(),
    ///             location.line(),
    ///         );
    ///     } else {
    ///         println!("panic occurred but can't get location information...");
    ///     }
    /// }));
    ///
    /// panic!("Normal panic");
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "panic_hooks", since = "1.10.0")]
    pub fn location(&self) -> Option<&Location<'_>> {
        // NOTE: If this is changed to sometimes return None,
        // deal with that case in std::panicking::default_hook and core::panicking::panic_fmt.
        Some(&self.location)
    }

    /// Returns whether the panic handler is allowed to unwind the stack from
    /// the point where the panic occurred.
    ///
    /// This is true for most kinds of panics with the exception of panics
    /// caused by trying to unwind out of a `Drop` implementation or a function
    /// whose ABI does not support unwinding.
    ///
    /// It is safe for a panic handler to unwind even when this function returns
    /// false, however this will simply cause the panic handler to be called
    /// again.
    #[must_use]
    #[inline]
    #[unstable(feature = "panic_can_unwind", issue = "92988")]
    pub fn can_unwind(&self) -> bool {
        self.can_unwind
    }

    #[unstable(
        feature = "panic_internals",
        reason = "internal details of the implementation of the `panic!` and related macros",
        issue = "none"
    )]
    #[doc(hidden)]
    #[inline]
    pub fn force_no_backtrace(&self) -> bool {
        self.force_no_backtrace
    }
}

#[stable(feature = "panic_hook_display", since = "1.26.0")]
impl fmt::Display for PanicHookInfo<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("panicked at ")?;
        self.location.fmt(formatter)?;
        if let Some(payload) = self.payload_as_str() {
            formatter.write_str(":\n")?;
            formatter.write_str(payload)?;
        }
        Ok(())
    }
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
#[allow_internal_unstable(libstd_sys_internals, const_format_args, panic_internals, rt)]
#[cfg_attr(not(test), rustc_diagnostic_item = "std_panic_2015_macro")]
#[rustc_macro_transparency = "semitransparent"]
pub macro panic_2015 {
    () => ({
        $crate::rt::begin_panic("explicit panic")
    }),
    ($msg:expr $(,)?) => ({
        $crate::rt::begin_panic($msg);
    }),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => ({
        $crate::rt::panic_display(&$arg);
    }),
    ($fmt:expr, $($arg:tt)+) => ({
        // Semicolon to prevent temporaries inside the formatting machinery from
        // being considered alive in the caller after the panic_fmt call.
        $crate::rt::panic_fmt($crate::const_format_args!($fmt, $($arg)+));
    }),
}

#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use core::panic::Location;
#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
pub use core::panic::panic_2021;
#[stable(feature = "catch_unwind", since = "1.9.0")]
pub use core::panic::{AssertUnwindSafe, RefUnwindSafe, UnwindSafe};

#[unstable(feature = "panic_update_hook", issue = "92649")]
pub use crate::panicking::update_hook;
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use crate::panicking::{set_hook, take_hook};

/// Panics the current thread with the given message as the panic payload.
///
/// The message can be of any (`Any + Send`) type, not just strings.
///
/// The message is wrapped in a `Box<'static + Any + Send>`, which can be
/// accessed later using [`PanicHookInfo::payload`].
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
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl UnwindSafe for Condvar {}

#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl<T: ?Sized> RefUnwindSafe for Mutex<T> {}
#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl<T: ?Sized> RefUnwindSafe for RwLock<T> {}
#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl RefUnwindSafe for Condvar {}

// https://github.com/rust-lang/rust/issues/62301
#[stable(feature = "hashbrown", since = "1.36.0")]
impl<K, V, S> UnwindSafe for collections::HashMap<K, V, S>
where
    K: UnwindSafe,
    V: UnwindSafe,
    S: UnwindSafe,
{
}

#[unstable(feature = "abort_unwind", issue = "130338")]
pub use core::panic::abort_unwind;

/// Invokes a closure, capturing the cause of an unwinding panic if one occurs.
///
/// This function will return `Ok` with the closure's result if the closure does
/// not panic, and will return `Err(cause)` if the closure panics. The `cause`
/// returned is the object with which panic was originally invoked.
///
/// Rust functions that are expected to be called from foreign code that does
/// not support unwinding (such as C compiled with `-fno-exceptions`) should be
/// defined using `extern "C"`, which ensures that if the Rust code panics, it
/// is automatically caught and the process is aborted. If this is the desired
/// behavior, it is not necessary to use `catch_unwind` explicitly. This
/// function should instead be used when more graceful error-handling is needed.
///
/// It is **not** recommended to use this function for a general try/catch
/// mechanism. The [`Result`] type is more appropriate to use for functions that
/// can fail on a regular basis. Additionally, this function is not guaranteed
/// to catch all panics, see the "Notes" section below.
///
/// The closure provided is required to adhere to the [`UnwindSafe`] trait to
/// ensure that all captured variables are safe to cross this boundary. The
/// purpose of this bound is to encode the concept of [exception safety][rfc] in
/// the type system. Most usage of this function should not need to worry about
/// this bound as programs are naturally unwind safe without `unsafe` code. If
/// it becomes a problem the [`AssertUnwindSafe`] wrapper struct can be used to
/// quickly assert that the usage here is indeed unwind safe.
///
/// [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1236-stabilize-catch-panic.md
///
/// # Notes
///
/// This function **might not catch all Rust panics**. A Rust panic is not
/// always implemented via unwinding, but can be implemented by aborting the
/// process as well. This function *only* catches unwinding panics, not those
/// that abort the process.
///
/// If a custom panic hook has been set, it will be invoked before the panic is
/// caught, before unwinding.
///
/// Although unwinding into Rust code with a foreign exception (e.g. an
/// exception thrown from C++ code, or a `panic!` in Rust code compiled or
/// linked with a different runtime) via an appropriate ABI (e.g. `"C-unwind"`)
/// is permitted, catching such an exception using this function will have one
/// of two behaviors, and it is unspecified which will occur:
///
/// * The process aborts, after executing all destructors of `f` and the
///   functions it called.
/// * The function returns a `Result::Err` containing an opaque type.
///
/// Finally, be **careful in how you drop the result of this function**. If it
/// is `Err`, it contains the panic payload, and dropping that may in turn
/// panic!
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

/// Makes all future panics abort directly without running the panic hook or unwinding.
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

    fn as_u8(self) -> u8 {
        match self {
            BacktraceStyle::Short => 1,
            BacktraceStyle::Full => 2,
            BacktraceStyle::Off => 3,
        }
    }

    fn from_u8(s: u8) -> Option<Self> {
        match s {
            1 => Some(BacktraceStyle::Short),
            2 => Some(BacktraceStyle::Full),
            3 => Some(BacktraceStyle::Off),
            _ => None,
        }
    }
}

// Tracks whether we should/can capture a backtrace, and how we should display
// that backtrace.
//
// Internally stores equivalent of an Option<BacktraceStyle>.
static SHOULD_CAPTURE: AtomicU8 = AtomicU8::new(0);

/// Configures whether the default panic hook will capture and display a
/// backtrace.
///
/// The default value for this setting may be set by the `RUST_BACKTRACE`
/// environment variable; see the details in [`get_backtrace_style`].
#[unstable(feature = "panic_backtrace_config", issue = "93346")]
pub fn set_backtrace_style(style: BacktraceStyle) {
    if cfg!(feature = "backtrace") {
        // If the `backtrace` feature of this crate is enabled, set the backtrace style.
        SHOULD_CAPTURE.store(style.as_u8(), Ordering::Relaxed);
    }
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

    let current = SHOULD_CAPTURE.load(Ordering::Relaxed);
    if let Some(style) = BacktraceStyle::from_u8(current) {
        return Some(style);
    }

    let format = match crate::env::var_os("RUST_BACKTRACE") {
        Some(x) if &x == "0" => BacktraceStyle::Off,
        Some(x) if &x == "full" => BacktraceStyle::Full,
        Some(_) => BacktraceStyle::Short,
        None if crate::sys::FULL_BACKTRACE_DEFAULT => BacktraceStyle::Full,
        None => BacktraceStyle::Off,
    };

    match SHOULD_CAPTURE.compare_exchange(0, format.as_u8(), Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => Some(format),
        Err(new) => BacktraceStyle::from_u8(new),
    }
}
