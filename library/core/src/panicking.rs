//! Panic support for libcore
//!
//! The core library cannot define panicking, but it does *declare* panicking. This
//! means that the functions inside of libcore are allowed to panic, but to be
//! useful an upstream crate must define panicking for libcore to use. The current
//! interface for panicking is:
//!
//! ```
//! fn panic_impl(pi: &core::panic::PanicInfo<'_>) -> !
//! # { loop {} }
//! ```
//!
//! This definition allows for panicking with any general message, but it does not
//! allow for failing with a `Box<Any>` value. (`PanicInfo` just contains a `&(dyn Any + Send)`,
//! for which we fill in a dummy value in `PanicInfo::internal_constructor`.)
//! The reason for this is that libcore is not allowed to allocate.
//!
//! This module contains a few other panicking functions, but these are just the
//! necessary lang items for the compiler. All panics are funneled through this
//! one function. The actual symbol is declared through the `#[panic_handler]` attribute.

#![allow(dead_code, missing_docs)]
#![unstable(
    feature = "core_panic",
    reason = "internal details of the implementation of the `panic!` and related macros",
    issue = "none"
)]

use crate::fmt;
use crate::panic::{Location, PanicInfo};

/// The underlying implementation of libcore's `panic!` macro when no formatting is used.
#[cold]
// never inline unless panic_immediate_abort to avoid code
// bloat at the call sites as much as possible
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[track_caller]
#[rustc_const_unstable(feature = "core_panic", issue = "none")]
#[lang = "panic"] // needed by codegen for panic on overflow and other `Assert` MIR terminators
pub const fn panic(expr: &'static str) -> ! {
    // Use Arguments::new_v1 instead of format_args!("{}", expr) to potentially
    // reduce size overhead. The format_args! macro uses str's Display trait to
    // write expr, which calls Formatter::pad, which must accommodate string
    // truncation and padding (even though none is used here). Using
    // Arguments::new_v1 may allow the compiler to omit Formatter::pad from the
    // output binary, saving up to a few kilobytes.
    panic_fmt(fmt::Arguments::new_v1(&[expr], &[]));
}

#[inline]
#[track_caller]
#[rustc_diagnostic_item = "panic_str"]
#[rustc_const_unstable(feature = "core_panic", issue = "none")]
pub const fn panic_str(expr: &str) -> ! {
    panic_display(&expr);
}

#[cfg(not(bootstrap))]
#[inline]
#[track_caller]
#[rustc_diagnostic_item = "unreachable_display"] // needed for `non-fmt-panics` lint
pub fn unreachable_display<T: fmt::Display>(x: &T) -> ! {
    panic_fmt(format_args!("internal error: entered unreachable code: {}", *x));
}

#[inline]
#[track_caller]
#[lang = "panic_display"] // needed for const-evaluated panics
#[rustc_do_not_const_check] // hooked by const-eval
#[rustc_const_unstable(feature = "core_panic", issue = "none")]
pub const fn panic_display<T: fmt::Display>(x: &T) -> ! {
    panic_fmt(format_args!("{}", *x));
}

#[cold]
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[track_caller]
#[lang = "panic_bounds_check"] // needed by codegen for panic on OOB array/slice access
fn panic_bounds_check(index: usize, len: usize) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        super::intrinsics::abort()
    }

    panic!("index out of bounds: the len is {} but the index is {}", len, index)
}

#[cfg(not(bootstrap))]
#[cold]
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[track_caller]
#[lang = "panic_no_unwind"] // needed by codegen for panic in nounwind function
fn panic_no_unwind() -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        super::intrinsics::abort()
    }

    // NOTE This function never crosses the FFI boundary; it's a Rust-to-Rust call
    // that gets resolved to the `#[panic_handler]` function.
    extern "Rust" {
        #[lang = "panic_impl"]
        fn panic_impl(pi: &PanicInfo<'_>) -> !;
    }

    // PanicInfo with the `can_unwind` flag set to false forces an abort.
    let fmt = format_args!("panic in a function that cannot unwind");
    let pi = PanicInfo::internal_constructor(Some(&fmt), Location::caller(), false);

    // SAFETY: `panic_impl` is defined in safe Rust code and thus is safe to call.
    unsafe { panic_impl(&pi) }
}

/// The entry point for panicking with a formatted message.
///
/// This is designed to reduce the amount of code required at the call
/// site as much as possible (so that `panic!()` has as low an impact
/// on (e.g.) the inlining of other functions as possible), by moving
/// the actual formatting into this shared place.
#[cold]
// If panic_immediate_abort, inline the abort call,
// otherwise avoid inlining because of it is cold path.
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[track_caller]
#[lang = "panic_fmt"] // needed for const-evaluated panics
#[rustc_do_not_const_check] // hooked by const-eval
#[rustc_const_unstable(feature = "core_panic", issue = "none")]
pub const fn panic_fmt(fmt: fmt::Arguments<'_>) -> ! {
    if cfg!(feature = "panic_immediate_abort") {
        super::intrinsics::abort()
    }

    // NOTE This function never crosses the FFI boundary; it's a Rust-to-Rust call
    // that gets resolved to the `#[panic_handler]` function.
    extern "Rust" {
        #[lang = "panic_impl"]
        fn panic_impl(pi: &PanicInfo<'_>) -> !;
    }

    let pi = PanicInfo::internal_constructor(Some(&fmt), Location::caller(), true);

    // SAFETY: `panic_impl` is defined in safe Rust code and thus is safe to call.
    unsafe { panic_impl(&pi) }
}

/// This function is used instead of panic_fmt in const eval.
#[lang = "const_panic_fmt"]
#[rustc_const_unstable(feature = "core_panic", issue = "none")]
pub const fn const_panic_fmt(fmt: fmt::Arguments<'_>) -> ! {
    if let Some(msg) = fmt.as_str() {
        panic_str(msg);
    } else {
        // SAFETY: This is only evaluated at compile time, which reliably
        // handles this UB (in case this branch turns out to be reachable
        // somehow).
        unsafe { crate::hint::unreachable_unchecked() };
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub enum AssertKind {
    Eq,
    Ne,
    Match,
}

/// Internal function for `assert_eq!` and `assert_ne!` macros
#[cold]
#[track_caller]
#[doc(hidden)]
pub fn assert_failed<T, U>(
    kind: AssertKind,
    left: &T,
    right: &U,
    args: Option<fmt::Arguments<'_>>,
) -> !
where
    T: fmt::Debug + ?Sized,
    U: fmt::Debug + ?Sized,
{
    assert_failed_inner(kind, &left, &right, args)
}

/// Internal function for `assert_match!`
#[cold]
#[track_caller]
#[doc(hidden)]
pub fn assert_matches_failed<T: fmt::Debug + ?Sized>(
    left: &T,
    right: &str,
    args: Option<fmt::Arguments<'_>>,
) -> ! {
    // Use the Display implementation to display the pattern.
    struct Pattern<'a>(&'a str);
    impl fmt::Debug for Pattern<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Display::fmt(self.0, f)
        }
    }
    assert_failed_inner(AssertKind::Match, &left, &Pattern(right), args);
}

/// Non-generic version of the above functions, to avoid code bloat.
#[track_caller]
fn assert_failed_inner(
    kind: AssertKind,
    left: &dyn fmt::Debug,
    right: &dyn fmt::Debug,
    args: Option<fmt::Arguments<'_>>,
) -> ! {
    let op = match kind {
        AssertKind::Eq => "==",
        AssertKind::Ne => "!=",
        AssertKind::Match => "matches",
    };

    match args {
        Some(args) => panic!(
            r#"assertion failed: `(left {} right)`
  left: `{:?}`,
 right: `{:?}`: {}"#,
            op, left, right, args
        ),
        None => panic!(
            r#"assertion failed: `(left {} right)`
  left: `{:?}`,
 right: `{:?}`"#,
            op, left, right,
        ),
    }
}
