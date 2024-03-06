//! Panic support in the standard library.

#![stable(feature = "core_panic_info", since = "1.41.0")]

mod location;
mod panic_info;
mod unwind_safe;

use crate::any::Any;

#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use self::location::Location;
#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use self::panic_info::PanicInfo;
#[stable(feature = "catch_unwind", since = "1.9.0")]
pub use self::unwind_safe::{AssertUnwindSafe, RefUnwindSafe, UnwindSafe};

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
    // Use `panic_str` instead of `panic_display::<&str>` for non_fmt_panic lint.
    ($msg:expr $(,)?) => ({
        $crate::panicking::panic_str($msg);
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

/// Like `assert_unsafe_precondition!` the defining features of this macro are that its
/// checks are enabled when they are monomorphized with debug assertions enabled, and upon failure
/// a non-unwinding panic is launched so that failures cannot compromise unwind safety.
///
/// But there are many differences from `assert_unsafe_precondition!`. This macro does not use
/// `const_eval_select` internally, and therefore it is sound to call this with an expression
/// that evaluates to `false`. Also unlike `assert_unsafe_precondition!` the condition being
/// checked here is not put in an outlined function. If the check compiles to a lot of IR, this
/// can cause code bloat if the check is monomorphized many times. But it also means that the checks
/// from this macro can be deduplicated or otherwise optimized out.
///
/// In general, this macro should be used to check all public-facing preconditions. But some
/// preconditions may be called too often or instantiated too often to make the overhead of the
/// checks tolerable. In such cases, place `#[cfg(debug_assertions)]` on the macro call. That will
/// disable the check in our precompiled standard library, but if a user wishes, they can still
/// enable the check by recompiling the standard library with debug assertions enabled.
#[doc(hidden)]
#[unstable(feature = "panic_internals", issue = "none")]
#[allow_internal_unstable(panic_internals, delayed_debug_assertions)]
#[rustc_macro_transparency = "semitransparent"]
pub macro debug_assert_nounwind {
    ($cond:expr $(,)?) => {
        if $crate::intrinsics::debug_assertions() {
            if !$cond {
                $crate::panicking::panic_nounwind($crate::concat!("assertion failed: ", $crate::stringify!($cond)));
            }
        }
    },
    ($cond:expr, $message:expr) => {
        if $crate::intrinsics::debug_assertions() {
            if !$cond {
                $crate::panicking::panic_nounwind($message);
            }
        }
    },
}

/// An internal trait used by std to pass data from std to `panic_unwind` and
/// other panic runtimes. Not intended to be stabilized any time soon, do not
/// use.
#[unstable(feature = "std_internals", issue = "none")]
#[doc(hidden)]
pub unsafe trait PanicPayload {
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
}
