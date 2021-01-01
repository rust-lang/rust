// These functions are used by macro expansion for bug! and span_bug!

use crate::ty::{tls, TyCtxt};
use rustc_span::{MultiSpan, Span};
use std::fmt;
use std::panic::Location;

#[cold]
#[inline(never)]
#[track_caller]
pub fn bug_fmt(args: fmt::Arguments<'_>) -> ! {
    // this wrapper mostly exists so I don't have to write a fully
    // qualified path of None::<Span> inside the bug!() macro definition
    opt_span_bug_fmt(None::<Span>, args, Location::caller());
}

#[cold]
#[inline(never)]
#[track_caller]
pub fn span_bug_fmt<S: Into<MultiSpan>>(span: S, args: fmt::Arguments<'_>) -> ! {
    opt_span_bug_fmt(Some(span), args, Location::caller());
}

#[track_caller]
fn opt_span_bug_fmt<S: Into<MultiSpan>>(
    span: Option<S>,
    args: fmt::Arguments<'_>,
    location: &Location<'_>,
) -> ! {
    tls::with_opt(move |tcx| {
        let msg = format!("{}: {}", location, args);
        match (tcx, span) {
            (Some(tcx), Some(span)) => tcx.sess.diagnostic().span_bug(span, &msg),
            (Some(tcx), None) => tcx.sess.diagnostic().bug(&msg),
            (None, _) => panic!(msg),
        }
    });
    unreachable!();
}

/// A query to trigger a `delay_span_bug`. Clearly, if one has a `tcx` one can already trigger a
/// `delay_span_bug`, so what is the point of this? It exists to help us test `delay_span_bug`'s
/// interactions with the query system and incremental.
pub fn trigger_delay_span_bug(tcx: TyCtxt<'_>, key: rustc_hir::def_id::DefId) {
    tcx.sess.delay_span_bug(
        tcx.def_span(key),
        "delayed span bug triggered by #[rustc_error(delay_span_bug_from_inside_query)]",
    );
}

pub fn provide(providers: &mut crate::ty::query::Providers) {
    *providers = crate::ty::query::Providers { trigger_delay_span_bug, ..*providers };
}
