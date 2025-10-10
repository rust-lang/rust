use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::LitKind;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Spanned;
use rustc_span::{Span, sym};

use super::VEC_RESIZE_TO_ZERO;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    count_arg: &'tcx Expr<'_>,
    default_arg: &'tcx Expr<'_>,
    name_span: Span,
) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
        && is_type_diagnostic_item(cx, cx.tcx.type_of(impl_id).instantiate_identity(), sym::Vec)
        && let ExprKind::Lit(Spanned {
            node: LitKind::Int(Pu128(0), _),
            ..
        }) = count_arg.kind
        && let ExprKind::Lit(Spanned {
            node: LitKind::Int(..), ..
        }) = default_arg.kind
    {
        let method_call_span = expr.span.with_lo(name_span.lo());
        span_lint_and_then(
            cx,
            VEC_RESIZE_TO_ZERO,
            expr.span,
            "emptying a vector with `resize`",
            |db| {
                db.help("the arguments may be inverted...");
                db.span_suggestion(
                    method_call_span,
                    "...or you can empty the vector with",
                    "clear()".to_string(),
                    Applicability::MaybeIncorrect,
                );
            },
        );
    }
}
