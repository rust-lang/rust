//! Panic support in the standard library.

#![stable(feature = "core_panic_info", since = "1.41.0")]

mod location;
mod panic_info;
mod unwind_safe;

#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use self::location::Location;
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use self::panic_info::PanicInfo;
#[stable(feature = "panic_info_message", since = "1.81.0")]
pub use self::panic_info::PanicMessage;
#[stable(feature = "catch_unwind", since = "1.9.0")]
pub use self::unwind_safe::{AssertUnwindSafe, RefUnwindSafe, UnwindSafe};
use crate::any::Any;

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
#[allow_internal_unstable(panic_internals, const_format_args)]
#[rustc_diagnostic_item = "core_panic_2015_macro"]
#[rustc_macro_transparency = "semitransparent"]
pub macro panic_2015 {
    () => (
        $crate::panicking::panic("explicit panic")
    ),
    ($msg:literal $(,)?) => (
        $crate::panicking::panic($msg)
    ),
    // Use `panic_str_2015` instead of `panic_display::<&str>` for non_fmt_panic lint.
    ($msg:expr $(,)?) => ({
        $crate::panicking::panic_str_2015($msg);
    }),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => ({
        $crate::panicking::panic_display(&$arg);
    }),
    ($fmt:expr, $($arg:tt)+) => ({
        // Semicolon to prevent temporaries inside the formatting machinery from
        // being considered alive in the caller after the panic_fmt call.
        $crate::panicking::panic_fmt($crate::const_format_args!($fmt, $($arg)+));
    }),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
#[allow_internal_unstable(panic_internals, const_format_args)]
#[rustc_diagnostic_item = "core_panic_2021_macro"]
#[rustc_macro_transparency = "semitransparent"]
#[cfg(feature = "panic_immediate_abort")]
pub macro panic_2021 {
    () => (
        $crate::panicking::panic("explicit panic")
    ),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => ({
        $crate::panicking::panic_display(&$arg);
    }),
    ($($t:tt)+) => ({
        // Semicolon to prevent temporaries inside the formatting machinery from
        // being considered alive in the caller after the panic_fmt call.
        $crate::panicking::panic_fmt($crate::const_format_args!($($t)+));
    }),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
#[allow_internal_unstable(
    panic_internals,
    core_intrinsics,
    const_dispatch,
    const_eval_select,
    const_format_args,
    rustc_attrs
)]
#[rustc_diagnostic_item = "core_panic_2021_macro"]
#[rustc_macro_transparency = "semitransparent"]
#[cfg(not(feature = "panic_immediate_abort"))]
pub macro panic_2021 {
    () => ({
        // Create a function so that the argument for `track_caller`
        // can be moved inside if possible.
        #[cold]
        #[track_caller]
        #[inline(never)]
        const fn panic_cold_explicit() -> ! {
            $crate::panicking::panic_explicit()
        }
        panic_cold_explicit();
    }),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => ({
        #[cold]
        #[track_caller]
        #[inline(never)]
        #[rustc_const_panic_str] // enforce a &&str argument in const-check and hook this by const-eval
        #[rustc_do_not_const_check] // hooked by const-eval
        const fn panic_cold_display<T: $crate::fmt::Display>(arg: &T) -> ! {
            $crate::panicking::panic_display(arg)
        }
        panic_cold_display(&$arg);
    }),
    ($($t:tt)+) => ({
        // Semicolon to prevent temporaries inside the formatting machinery from
        // being considered alive in the caller after the panic_fmt call.
        $crate::panicking::panic_fmt($crate::const_format_args!($($t)+));
    }),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use unreachable!() instead")]
#[allow_internal_unstable(panic_internals)]
#[rustc_diagnostic_item = "unreachable_2015_macro"]
#[rustc_macro_transparency = "semitransparent"]
pub macro unreachable_2015 {
    () => (
        $crate::panicking::panic("internal error: entered unreachable code")
    ),
    // Use of `unreachable_display` for non_fmt_panic lint.
    // NOTE: the message ("internal error ...") is embedded directly in unreachable_display
    ($msg:expr $(,)?) => ({
        $crate::panicking::unreachable_display(&$msg);
    }),
    ($fmt:expr, $($arg:tt)*) => (
        $crate::panic!($crate::concat!("internal error: entered unreachable code: ", $fmt), $($arg)*)
    ),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use unreachable!() instead")]
#[allow_internal_unstable(panic_internals)]
#[rustc_macro_transparency = "semitransparent"]
pub macro unreachable_2021 {
    () => (
        $crate::panicking::panic("internal error: entered unreachable code")
    ),
    ($($t:tt)+) => (
        $crate::panic!("internal error: entered unreachable code: {}", $crate::format_args!($($t)+))
    ),
}

/// Invokes a closure, aborting if the closure unwinds.
///
/// When compiled with aborting panics, this function is effectively a no-op.
/// With unwinding panics, an unwind results in another call into the panic
/// hook followed by a process abort.
///
/// # Notes
///
/// Instead of using this function, code should attempt to support unwinding.
/// Implementing [`Drop`] allows you to restore invariants uniformly in both
/// return and unwind paths.
///
/// If an unwind can lead to logical issues but not soundness issues, you
/// should allow the unwind. Opting out of [`UnwindSafe`] indicates to your
/// consumers that they need to consider correctness in the face of unwinds.
///
/// If an unwind would be unsound, then this function should be used in order
/// to prevent unwinds. However, note that `extern "C" fn` will automatically
/// convert unwinds to aborts, so using this function isn't necessary for FFI.
#[unstable(feature = "abort_unwind", issue = "130338")]
#[rustc_nounwind]
pub fn abort_unwind<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

/// An internal trait used by std to pass data from std to `panic_unwind` and
/// other panic runtimes. Not intended to be stabilized any time soon, do not
/// use.
#[unstable(feature = "std_internals", issue = "none")]
#[doc(hidden)]
pub unsafe trait PanicPayload: crate::fmt::Display {
    /// Take full ownership of the contents.
    /// The return type is actually `Box<dyn Any + Send>`, but we cannot use `Box` in core.
    ///
    /// After this method got called, only some dummy default value is left in `self`.
    /// Calling this method twice, or calling `get` after calling this method, is an error.
    ///
    /// The argument is borrowed because the panic runtime (`__rust_start_panic`) only
    /// gets a borrowed `dyn PanicPayload`.
    fn take_box(&mut self) -> *mut (dyn Any + Send);

    /// Just borrow the contents.
    fn get(&mut self) -> &(dyn Any + Send);

    /// Tries to borrow the contents as `&str`, if possible without doing any allocations.
    fn as_str(&mut self) -> Option<&str> {
        None
    }
}

/// Helper macro for panicking in a `const fn`.
/// Invoke as:
/// ```rust,ignore (just an example)
/// core::macros::const_panic!("boring message", "flavored message {a} {b:?}", a: u32 = foo.len(), b: Something = bar);
/// ```
/// where the first message will be printed in const-eval,
/// and the second message will be printed at runtime.
// All uses of this macro are FIXME(const-hack).
#[unstable(feature = "panic_internals", issue = "none")]
#[doc(hidden)]
pub macro const_panic {
    ($const_msg:literal, $runtime_msg:literal, $($arg:ident : $ty:ty = $val:expr),* $(,)?) => {{
        // Wrap call to `const_eval_select` in a function so that we can
        // add the `rustc_allow_const_fn_unstable`. This is okay to do
        // because both variants will panic, just with different messages.
        #[rustc_allow_const_fn_unstable(const_eval_select)]
        #[inline(always)] // inline the wrapper
        #[track_caller]
        #[cfg_attr(bootstrap, rustc_const_stable(feature = "const_panic", since = "1.84.0"))]
        const fn do_panic($($arg: $ty),*) -> ! {
            $crate::intrinsics::const_eval_select!(
                @capture { $($arg: $ty = $arg),* } -> !:
                #[noinline]
                if const #[track_caller] #[inline] { // Inline this, to prevent codegen
                    $crate::panic!($const_msg)
                } else #[track_caller] #[cfg_attr(bootstrap, inline)] { // Do not inline this, it makes perf worse
                    $crate::panic!($runtime_msg)
                }
            )
        }

        do_panic($($val),*)
    }},
    // We support leaving away the `val` expressions for *all* arguments
    // (but not for *some* arguments, that's too tricky).
    ($const_msg:literal, $runtime_msg:literal, $($arg:ident : $ty:ty),* $(,)?) => {
        $crate::panic::const_panic!(
            $const_msg,
            $runtime_msg,
            $($arg: $ty = $arg),*
        )
    },
}

/// A version of `assert` that prints a non-formatting message in const contexts.
///
/// See [`const_panic!`].
#[unstable(feature = "panic_internals", issue = "none")]
#[doc(hidden)]
pub macro const_assert {
    ($condition: expr, $const_msg:literal, $runtime_msg:literal, $($arg:tt)*) => {{
        if !$crate::intrinsics::likely($condition) {
            $crate::panic::const_panic!($const_msg, $runtime_msg, $($arg)*)
        }
    }}
}
