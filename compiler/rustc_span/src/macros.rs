use std::fmt;
use std::panic::Location;

use rustc_data_structures::AtomicRef;

use crate::Span;

/// A macro for triggering an ICE.
/// Calling `bug` instead of panicking will result in a nicer error message and should
/// therefore be preferred over `panic`/`unreachable` or others.
///
/// If you have a span available, you should use [`span_bug`] instead.
///
/// If the bug should only be emitted when compilation didn't fail,
/// [`DiagCtxtHandle::span_delayed_bug`] may be useful.
///
/// [`DiagCtxtHandle::span_delayed_bug`]: ../rustc_errors/struct.DiagCtxtHandle.html#method.span_delayed_bug
/// [`span_bug`]: crate::span_bug
pub macro bug {
    () => (
        bug!("impossible case reached")
    ),
    ($($arg:tt)+) => (
        bug_impl(None, std::format_args!($($arg)+), Location::caller())
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
/// [`DiagCtxtHandle::span_delayed_bug`]: ../rustc_errors/struct.DiagCtxtHandle.html#method.span_delayed_bug
pub macro span_bug($span:expr, $($arg:tt)+){
   bug_impl(Some($span), std::format_args!($($arg)+), Location::caller())
}

#[cold]
#[track_caller]
pub fn bug_impl(span: Option<Span>, args: fmt::Arguments<'_>, location: &Location<'_>) -> ! {
    (*EMIT_BUG_DIAGNOSTIC)(span, args, location);
    panic!("{args}")
}

pub static EMIT_BUG_DIAGNOSTIC: AtomicRef<fn(Option<Span>, fmt::Arguments<'_>, &Location<'_>)> =
    AtomicRef::new(&(default_emit_diagnostic as _));

fn default_emit_diagnostic(_: Option<Span>, _: fmt::Arguments<'_>, _: &Location<'_>) {}
