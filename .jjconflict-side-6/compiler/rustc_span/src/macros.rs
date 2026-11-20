use std::fmt;
use std::panic::{Location, panic_any};

use rustc_data_structures::AtomicRef;

use crate::Span;

/// Signifies that the compiler died with an explicit call to `.bug` or `.span_bug` rather than a
/// failed assertion, etc.
pub struct ExplicitBug;

/// A macro for triggering an ICE.
/// Calling `bug` instead of panicking will result in a nicer error message and should
/// therefore be preferred over `panic`/`unreachable` or others.
///
/// If you have a span available, you should use [`span_bug`] instead.
///
/// If the bug should only be emitted when compilation didn't fail,
/// [`DiagCtxtHandle::span_delayed_bug`] may be useful.
///
/// [`DiagCtxtHandle::span_delayed_bug`]: ../../rustc_errors/struct.DiagCtxtHandle.html#method.span_delayed_bug
/// [`span_bug`]: crate::span_bug
pub macro bug {
    () => (
        bug!("impossible case reached")
    ),
    ($($arg:tt)+) => (
        // Use the full path of `bug_impl` to make sure rust-analyzer can resolve it,
        // to avoid bogus type errors about `()` versus `!`.
        $crate::macros::bug_impl(None, std::format_args!($($arg)+), Location::caller())
    ),
}

/// A macro for triggering an ICE with a span.
/// Calling `span_bug!` instead of panicking will result in a nicer error message and point
/// at the code the compiler was compiling when it ICEd. This is the preferred way to trigger
/// ICEs.
///
/// If the bug should only be emitted when compilation didn't fail,
/// [`DiagCtxtHandle::span_delayed_bug`] may be useful.
///
/// [`DiagCtxtHandle::span_delayed_bug`]: ../../rustc_errors/struct.DiagCtxtHandle.html#method.span_delayed_bug
pub macro span_bug($span:expr, $($arg:tt)+){
    // Use the full path of `bug_impl` to make sure rust-analyzer can resolve it,
    // to avoid bogus type errors about `()` versus `!`.
    $crate::macros::bug_impl(Some($span), std::format_args!($($arg)+), Location::caller())
}

#[cold]
#[track_caller]
pub fn bug_impl(
    span: Option<Span>,
    args: fmt::Arguments<'_>,
    location: &'static Location<'static>,
) -> ! {
    // Emit the bug without aborting.
    let emitted = (*EMIT_BUG_DIAGNOSTIC)(span, args, location);

    if emitted {
        // Panic with `ExplicitBug`, which tells `report_ice` that it's expected, e.g. originating
        // from `bug!` or `dcx.emit_bug(..)`.
        panic_any(ExplicitBug);
    } else {
        // Panic with just a string, which means it's unexpected.
        panic_any(format!("{args}"));
    }
}

pub static EMIT_BUG_DIAGNOSTIC: AtomicRef<
    fn(Option<Span>, fmt::Arguments<'_>, &'static Location<'static>) -> bool,
> = AtomicRef::new(&(default_emit_bug_diagnostic as _));

fn default_emit_bug_diagnostic(
    _: Option<Span>,
    _args: fmt::Arguments<'_>,
    _location: &'static Location<'static>,
) -> bool {
    false
}
