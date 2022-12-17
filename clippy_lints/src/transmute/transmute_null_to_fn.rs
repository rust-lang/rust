use clippy_utils::consts::{constant_context, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{is_integer_literal, is_path_diagnostic_item};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::symbol::sym;

use super::TRANSMUTE_NULL_TO_FN;

const LINT_MSG: &str = "transmuting a known null pointer into a function pointer";
const NOTE_MSG: &str = "this transmute results in a null function pointer";
const HELP_MSG: &str =
    "try wrapping your function pointer type in `Option<T>` instead, and using `None` as a null pointer value";

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arg: &'tcx Expr<'_>, to_ty: Ty<'tcx>) -> bool {
    if !to_ty.is_fn() {
        return false;
    }

    // Catching transmute over constants that resolve to `null`.
    let mut const_eval_context = constant_context(cx, cx.typeck_results());
    if let ExprKind::Path(ref _qpath) = arg.kind &&
        let Some(Constant::RawPtr(x)) = const_eval_context.expr(arg) &&
        x == 0
    {
        span_lint_and_then(cx, TRANSMUTE_NULL_TO_FN, expr.span, LINT_MSG, |diag| {
            diag.span_label(expr.span, NOTE_MSG);
            diag.help(HELP_MSG);
        });
        return true;
    }

    // Catching:
    // `std::mem::transmute(0 as *const i32)`
    if let ExprKind::Cast(inner_expr, _cast_ty) = arg.kind && is_integer_literal(inner_expr, 0) {
        span_lint_and_then(cx, TRANSMUTE_NULL_TO_FN, expr.span, LINT_MSG, |diag| {
            diag.span_label(expr.span, NOTE_MSG);
            diag.help(HELP_MSG);
        });
        return true;
    }

    // Catching:
    // `std::mem::transmute(std::ptr::null::<i32>())`
    if let ExprKind::Call(func1, []) = arg.kind &&
        is_path_diagnostic_item(cx, func1, sym::ptr_null)
    {
        span_lint_and_then(cx, TRANSMUTE_NULL_TO_FN, expr.span, LINT_MSG, |diag| {
            diag.span_label(expr.span, NOTE_MSG);
            diag.help(HELP_MSG);
        });
        return true;
    }

    // FIXME:
    // Also catch transmutations of variables which are known nulls.
    // To do this, MIR const propagation seems to be the better tool.
    // Whenever MIR const prop routines are more developed, this will
    // become available. As of this writing (25/03/19) it is not yet.
    false
}
