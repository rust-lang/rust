use clippy_utils::consts::{constant_context, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_path_diagnostic_item;
use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::symbol::sym;

use super::TRANSMUTING_NULL;

const LINT_MSG: &str = "transmuting a known null pointer into a reference";

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arg: &'tcx Expr<'_>, to_ty: Ty<'tcx>) -> bool {
    if !to_ty.is_ref() {
        return false;
    }

    // Catching transmute over constants that resolve to `null`.
    let mut const_eval_context = constant_context(cx, cx.typeck_results());
    if_chain! {
        if let ExprKind::Path(ref _qpath) = arg.kind;
        if let Some(Constant::RawPtr(x)) = const_eval_context.expr(arg);
        if x == 0;
        then {
            span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG);
            return true;
        }
    }

    // Catching:
    // `std::mem::transmute(0 as *const i32)`
    if_chain! {
        if let ExprKind::Cast(inner_expr, _cast_ty) = arg.kind;
        if let ExprKind::Lit(ref lit) = inner_expr.kind;
        if let LitKind::Int(0, _) = lit.node;
        then {
            span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG);
            return true;
        }
    }

    // Catching:
    // `std::mem::transmute(std::ptr::null::<i32>())`
    if_chain! {
        if let ExprKind::Call(func1, []) = arg.kind;
        if is_path_diagnostic_item(cx, func1, sym::ptr_null);
        then {
            span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG);
            return true;
        }
    }

    // FIXME:
    // Also catch transmutations of variables which are known nulls.
    // To do this, MIR const propagation seems to be the better tool.
    // Whenever MIR const prop routines are more developed, this will
    // become available. As of this writing (25/03/19) it is not yet.
    false
}
