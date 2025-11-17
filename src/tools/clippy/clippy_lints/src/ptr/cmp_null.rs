use super::CMP_NULL;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef, MaybeResPath};
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_lint_allowed, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    op: BinOpKind,
    l: &Expr<'_>,
    r: &Expr<'_>,
) -> bool {
    let non_null_path_snippet = match (
        is_lint_allowed(cx, CMP_NULL, expr.hir_id),
        is_null_path(cx, l),
        is_null_path(cx, r),
    ) {
        (false, true, false) if let Some(sugg) = Sugg::hir_opt(cx, r) => sugg.maybe_paren(),
        (false, false, true) if let Some(sugg) = Sugg::hir_opt(cx, l) => sugg.maybe_paren(),
        _ => return false,
    };
    let invert = if op == BinOpKind::Eq { "" } else { "!" };

    span_lint_and_sugg(
        cx,
        CMP_NULL,
        expr.span,
        "comparing with null is better expressed by the `.is_null()` method",
        "try",
        format!("{invert}{non_null_path_snippet}.is_null()",),
        Applicability::MachineApplicable,
    );
    true
}

fn is_null_path(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::Call(pathexp, []) = expr.kind {
        matches!(
            pathexp.basic_res().opt_diag_name(cx),
            Some(sym::ptr_null | sym::ptr_null_mut)
        )
    } else {
        false
    }
}
