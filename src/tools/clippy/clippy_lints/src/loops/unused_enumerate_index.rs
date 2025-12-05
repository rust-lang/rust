use super::UNUSED_ENUMERATE_INDEX;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::{SpanRangeExt, walk_span_to_context};
use clippy_utils::{expr_or_init, pat_is_wild};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Pat, PatKind, TyKind};
use rustc_lint::LateContext;
use rustc_span::{Span, SyntaxContext, sym};

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    iter_expr: &'tcx Expr<'tcx>,
    pat: &Pat<'tcx>,
    ty_spans: Option<(Span, Span)>,
    body: &'tcx Expr<'tcx>,
) {
    if let PatKind::Tuple([idx_pat, inner_pat], _) = pat.kind
        && cx.typeck_results().expr_ty(iter_expr).is_diag_item(cx, sym::Enumerate)
        && pat_is_wild(cx, &idx_pat.kind, body)
        && let enumerate_call = expr_or_init(cx, iter_expr)
        && let ExprKind::MethodCall(_, _, [], enumerate_span) = enumerate_call.kind
        && let Some(enumerate_id) = cx.typeck_results().type_dependent_def_id(enumerate_call.hir_id)
        && cx.tcx.is_diagnostic_item(sym::enumerate_method, enumerate_id)
        && !enumerate_call.span.from_expansion()
        && !pat.span.from_expansion()
        && !idx_pat.span.from_expansion()
        && !inner_pat.span.from_expansion()
        && let Some(enumerate_range) = enumerate_span.map_range(cx, |_, text, range| {
            text.get(..range.start)?
                .ends_with('.')
                .then_some(range.start - 1..range.end)
        })
    {
        let enumerate_span = Span::new(enumerate_range.start, enumerate_range.end, SyntaxContext::root(), None);
        span_lint_hir_and_then(
            cx,
            UNUSED_ENUMERATE_INDEX,
            enumerate_call.hir_id,
            enumerate_span,
            "you seem to use `.enumerate()` and immediately discard the index",
            |diag| {
                let mut spans = Vec::with_capacity(5);
                spans.push((enumerate_span, String::new()));
                spans.push((pat.span.with_hi(inner_pat.span.lo()), String::new()));
                spans.push((pat.span.with_lo(inner_pat.span.hi()), String::new()));
                if let Some((outer, inner)) = ty_spans {
                    spans.push((outer.with_hi(inner.lo()), String::new()));
                    spans.push((outer.with_lo(inner.hi()), String::new()));
                }
                diag.multipart_suggestion(
                    "remove the `.enumerate()` call",
                    spans,
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}

pub(super) fn check_method<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    arg: &'tcx Expr<'tcx>,
) {
    if let ExprKind::Closure(closure) = arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && let [param] = body.params
        && cx.ty_based_def(e).opt_parent(cx).is_diag_item(cx, sym::Iterator)
        && let [input] = closure.fn_decl.inputs
        && !arg.span.from_expansion()
        && !input.span.from_expansion()
        && !recv.span.from_expansion()
        && !param.span.from_expansion()
    {
        let ty_spans = if let TyKind::Tup([_, inner]) = input.kind {
            let Some(inner) = walk_span_to_context(inner.span, SyntaxContext::root()) else {
                return;
            };
            Some((input.span, inner))
        } else {
            None
        };
        check(cx, recv, param.pat, ty_spans, body.value);
    }
}
