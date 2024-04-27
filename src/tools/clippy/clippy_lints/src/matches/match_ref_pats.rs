use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, walk_span_to_context};
use clippy_utils::sugg::Sugg;
use core::iter::once;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, Pat, PatKind};
use rustc_lint::LateContext;

use super::MATCH_REF_PATS;

pub(crate) fn check<'a, 'b, I>(cx: &LateContext<'_>, scrutinee: &Expr<'_>, pats: I, expr: &Expr<'_>)
where
    'b: 'a,
    I: Clone + Iterator<Item = &'a Pat<'b>>,
{
    if !has_multiple_ref_pats(pats.clone()) {
        return;
    }

    let (first_sugg, msg, title);
    let ctxt = expr.span.ctxt();
    let mut app = Applicability::Unspecified;
    if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner) = scrutinee.kind {
        if scrutinee.span.ctxt() != ctxt {
            return;
        }
        first_sugg = once((
            scrutinee.span,
            Sugg::hir_with_context(cx, inner, ctxt, "..", &mut app).to_string(),
        ));
        msg = "try";
        title = "you don't need to add `&` to both the expression and the patterns";
    } else {
        let Some(span) = walk_span_to_context(scrutinee.span, ctxt) else {
            return;
        };
        first_sugg = once((
            span,
            Sugg::hir_with_context(cx, scrutinee, ctxt, "..", &mut app)
                .deref()
                .to_string(),
        ));
        msg = "instead of prefixing all patterns with `&`, you can dereference the expression";
        title = "you don't need to add `&` to all patterns";
    }

    let remaining_suggs = pats.filter_map(|pat| {
        if let PatKind::Ref(refp, _) = pat.kind {
            Some((pat.span, snippet(cx, refp.span, "..").to_string()))
        } else {
            None
        }
    });

    span_lint_and_then(cx, MATCH_REF_PATS, expr.span, title, |diag| {
        if !expr.span.from_expansion() {
            multispan_sugg(diag, msg, first_sugg.chain(remaining_suggs));
        }
    });
}

fn has_multiple_ref_pats<'a, 'b, I>(pats: I) -> bool
where
    'b: 'a,
    I: Iterator<Item = &'a Pat<'b>>,
{
    let mut ref_count = 0;
    for opt in pats.map(|pat| match pat.kind {
        PatKind::Ref(..) => Some(true), // &-patterns
        PatKind::Wild => Some(false),   // an "anything" wildcard is also fine
        _ => None,                      // any other pattern is not fine
    }) {
        if let Some(inner) = opt {
            if inner {
                ref_count += 1;
            }
        } else {
            return false;
        }
    }
    ref_count > 1
}
