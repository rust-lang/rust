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
#[allow_internal_unstable(core_panic, const_format_args)]
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
    ($msg:expr $(,)?) => (
        $crate::panicking::panic_str($msg)
    ),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => (
        $crate::panicking::panic_display(&$arg)
    ),
    ($fmt:expr, $($arg:tt)+) => (
        $crate::panicking::panic_fmt($crate::const_format_args!($fmt, $($arg)+))
    ),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use panic!() instead")]
#[allow_internal_unstable(core_panic, const_format_args)]
#[rustc_diagnostic_item = "core_panic_2021_macro"]
#[rustc_macro_transparency = "semitransparent"]
pub macro panic_2021 {
    () => (
        $crate::panicking::panic("explicit panic")
    ),
    // Special-case the single-argument case for const_panic.
    ("{}", $arg:expr $(,)?) => (
        $crate::panicking::panic_display(&$arg)
    ),
    ($($t:tt)+) => (
        $crate::panicking::panic_fmt($crate::const_format_args!($($t)+))
    ),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use unreachable!() instead")]
#[allow_internal_unstable(core_panic)]
#[rustc_diagnostic_item = "unreachable_2015_macro"]
#[rustc_macro_transparency = "semitransparent"]
pub macro unreachable_2015 {
    () => (
        $crate::panicking::panic("internal error: entered unreachable code")
    ),
    // Use of `unreachable_display` for non_fmt_panic lint.
    // NOTE: the message ("internal error ...") is embeded directly in unreachable_display
    ($msg:expr $(,)?) => (
        $crate::panicking::unreachable_display(&$msg)
    ),
    ($fmt:expr, $($arg:tt)*) => (
        $crate::panic!($crate::concat!("internal error: entered unreachable code: ", $fmt), $($arg)*)
    ),
}

#[doc(hidden)]
#[unstable(feature = "edition_panic", issue = "none", reason = "use unreachable!() instead")]
#[allow_internal_unstable(core_panic)]
#[rustc_diagnostic_item = "unreachable_2021_macro"]
#[rustc_macro_transparency = "semitransparent"]
pub macro unreachable_2021 {
    () => (
        $crate::panicking::panic("internal error: entered unreachable code")
    ),
    ($($t:tt)+) => (
        $crate::panic!("internal error: entered unreachable code: {}", $crate::format_args!($($t)+))
    ),
}

/// An internal trait used by libstd to pass data from libstd to `panic_unwind`
/// and other panic runtimes. Not intended to be stabilized any time soon, do
/// not use.
#[unstable(feature = "std_internals", issue = "none")]
#[doc(hidden)]
pub unsafe trait BoxMeUp {
    /// Take full ownership of the contents.
    /// The return type is actually `Box<dyn Any + Send>`, but we cannot use `Box` in libcore.
    ///
    /// After this method got called, only some dummy default value is left in `self`.
    /// Calling this method twice, or calling `get` after calling this method, is an error.
    ///
    /// The argument is borrowed because the panic runtime (`__rust_start_panic`) only
    /// gets a borrowed `dyn BoxMeUp`.
    fn take_box(&mut self) -> *mut (dyn Any + Send);

    /// Just borrow the contents.
    fn get(&mut self) -> &(dyn Any + Send);
}
