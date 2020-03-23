// These functions are used by macro expansion for bug! and span_bug!

use crate::ty::{tls, TyCtxt};
use rustc_span::{MultiSpan, Span};
use std::fmt;

#[cold]
#[inline(never)]
pub fn bug_fmt(file: &'static str, line: u32, args: fmt::Arguments<'_>) -> ! {
    // this wrapper mostly exists so I don't have to write a fully
    // qualified path of None::<Span> inside the bug!() macro definition
    opt_span_bug_fmt(file, line, None::<Span>, args);
}

#[cold]
#[inline(never)]
pub fn span_bug_fmt<S: Into<MultiSpan>>(
    file: &'static str,
    line: u32,
    span: S,
    args: fmt::Arguments<'_>,
) -> ! {
    opt_span_bug_fmt(file, line, Some(span), args);
}

fn opt_span_bug_fmt<S: Into<MultiSpan>>(
    file: &'static str,
    line: u32,
    span: Option<S>,
    args: fmt::Arguments<'_>,
) -> ! {
    tls::with_opt(move |tcx| {
        let msg = format!("{}:{}: {}", file, line, args);
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

pub fn provide(providers: &mut crate::ty::query::Providers<'_>) {
    *providers = crate::ty::query::Providers { trigger_delay_span_bug, ..*providers };
}
