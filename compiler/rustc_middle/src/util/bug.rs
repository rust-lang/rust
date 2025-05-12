// These functions are used by macro expansion for `bug!` and `span_bug!`.

use std::fmt;
use std::panic::{Location, panic_any};

use rustc_errors::MultiSpan;
use rustc_span::Span;

use crate::ty::{TyCtxt, tls};

// This wrapper makes for more compact code at callsites than calling `opt_span_buf_fmt` directly.
#[cold]
#[inline(never)]
#[track_caller]
pub fn bug_fmt(args: fmt::Arguments<'_>) -> ! {
    opt_span_bug_fmt(None::<Span>, args, Location::caller());
}

// This wrapper makes for more compact code at callsites than calling `opt_span_buf_fmt` directly.
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
    tls::with_opt(
        #[track_caller]
        move |tcx| {
            let msg = format!("{location}: {args}");
            match (tcx, span) {
                (Some(tcx), Some(span)) => tcx.dcx().span_bug(span, msg),
                (Some(tcx), None) => tcx.dcx().bug(msg),
                (None, _) => panic_any(msg),
            }
        },
    )
}

/// A query to trigger a delayed bug. Clearly, if one has a `tcx` one can already trigger a
/// delayed bug, so what is the point of this? It exists to help us test the interaction of delayed
/// bugs with the query system and incremental.
pub fn trigger_delayed_bug(tcx: TyCtxt<'_>, key: rustc_hir::def_id::DefId) {
    tcx.dcx().span_delayed_bug(
        tcx.def_span(key),
        "delayed bug triggered by #[rustc_delayed_bug_from_inside_query]",
    );
}

pub fn provide(providers: &mut crate::query::Providers) {
    *providers = crate::query::Providers { trigger_delayed_bug, ..*providers };
}
