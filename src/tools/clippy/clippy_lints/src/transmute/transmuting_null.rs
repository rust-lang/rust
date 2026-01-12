use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_integer_const;
use clippy_utils::res::{MaybeDef, MaybeResPath};
use rustc_hir::{ConstBlock, Expr, ExprKind};
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
    if let ExprKind::Path(ref _qpath) = arg.kind
        && let Some(Constant::RawPtr(0)) = ConstEvalCtxt::new(cx).eval(arg)
    {
        span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG);
        return true;
    }

    // Catching:
    // `std::mem::transmute(0 as *const i32)`
    if let ExprKind::Cast(inner_expr, _cast_ty) = arg.kind
        && is_integer_const(cx, inner_expr, 0)
    {
        span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG);
        return true;
    }

    // Catching:
    // `std::mem::transmute(std::ptr::null::<i32>())`
    if let ExprKind::Call(func1, []) = arg.kind
        && func1.basic_res().is_diag_item(cx, sym::ptr_null)
    {
        span_lint(cx, TRANSMUTING_NULL, expr.span, LINT_MSG);
        return true;
    }

    // Catching:
    // `std::mem::transmute({ 0 as *const u64 })` and similar const blocks
    if let ExprKind::Block(block, _) = arg.kind
        && block.stmts.is_empty()
        && let Some(inner) = block.expr
    {
        // Run again with the inner expression
        return check(cx, expr, inner, to_ty);
    }

    // Catching:
    // `std::mem::transmute(const { u64::MIN as *const u64 });`
    if let ExprKind::ConstBlock(ConstBlock { body, .. }) = arg.kind {
        // Strip out the const and run again
        let block = cx.tcx.hir_body(body).value;
        return check(cx, expr, block, to_ty);
    }

    false
}
