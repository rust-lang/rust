use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_integer_literal, is_path_diagnostic_item};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::symbol::sym;

use super::TRANSMUTE_NULL_TO_FN;

fn lint_expr(cx: &LateContext<'_>, expr: &Expr<'_>) {
    span_lint_and_then(
        cx,
        TRANSMUTE_NULL_TO_FN,
        expr.span,
        "transmuting a known null pointer into a function pointer",
        |diag| {
            diag.span_label(expr.span, "this transmute results in undefined behavior");
            diag.help(
               "try wrapping your function pointer type in `Option<T>` instead, and using `None` as a null pointer value"
            );
        },
    );
}

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arg: &'tcx Expr<'_>, to_ty: Ty<'tcx>) -> bool {
    if !to_ty.is_fn() {
        return false;
    }

    let casts_peeled = peel_casts(arg);
    match casts_peeled.kind {
        // Catching:
        // transmute over constants that resolve to `null`.
        ExprKind::Path(ref _qpath)
            if matches!(ConstEvalCtxt::new(cx).eval(casts_peeled), Some(Constant::RawPtr(0))) =>
        {
            lint_expr(cx, expr);
            true
        },
        // Catching:
        // `std::mem::transmute(std::ptr::null::<i32>())`
        ExprKind::Call(func1, []) if is_path_diagnostic_item(cx, func1, sym::ptr_null) => {
            lint_expr(cx, expr);
            true
        },
        _ => {
            // FIXME:
            // Also catch transmutations of variables which are known nulls.
            // To do this, MIR const propagation seems to be the better tool.
            // Whenever MIR const prop routines are more developed, this will
            // become available. As of this writing (25/03/19) it is not yet.
            if is_integer_literal(casts_peeled, 0) {
                lint_expr(cx, expr);
                return true;
            }
            false
        },
    }
}

fn peel_casts<'tcx>(expr: &'tcx Expr<'tcx>) -> &'tcx Expr<'tcx> {
    match &expr.kind {
        ExprKind::Cast(inner_expr, _) => peel_casts(inner_expr),
        _ => expr,
    }
}
