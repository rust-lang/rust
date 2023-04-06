use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_range_full;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use rustc_span::Span;

use super::CLEAR_WITH_DRAIN;

const ACCEPTABLE_TYPES_WITH_ARG: [rustc_span::Symbol; 3] = [sym::String, sym::Vec, sym::VecDeque];

const ACCEPTABLE_TYPES_WITHOUT_ARG: [rustc_span::Symbol; 3] = [sym::BinaryHeap, sym::HashMap, sym::HashSet];

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span, arg: Option<&Expr<'_>>) {
    if let Some(arg) = arg {
        if match_acceptable_type(cx, recv, &ACCEPTABLE_TYPES_WITH_ARG)
            && let ExprKind::Path(QPath::Resolved(None, container_path)) = recv.kind
            && is_range_full(cx, arg, Some(container_path))
        {
            suggest(cx, expr, recv, span);
        }
    } else if match_acceptable_type(cx, recv, &ACCEPTABLE_TYPES_WITHOUT_ARG) {
        suggest(cx, expr, recv, span);
    }
}

fn match_acceptable_type(cx: &LateContext<'_>, expr: &hir::Expr<'_>, types: &[rustc_span::Symbol]) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr).peel_refs();
    types.iter().any(|&ty| is_type_diagnostic_item(cx, expr_ty, ty))
}

fn suggest(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, span: Span) {
    if let Some(adt) = cx.typeck_results().expr_ty(recv).ty_adt_def()
        && let Some(ty_name) = cx.tcx.get_diagnostic_name(adt.did())
    {
        span_lint_and_sugg(
            cx,
            CLEAR_WITH_DRAIN,
            span.with_hi(expr.span.hi()),
            &format!("`drain` used to clear a `{}`", ty_name),
            "try",
            "clear()".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
