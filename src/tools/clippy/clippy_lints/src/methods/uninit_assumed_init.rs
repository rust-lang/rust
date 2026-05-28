use clippy_utils::diagnostics::span_lint;
use clippy_utils::res::{MaybeDef, MaybeQPath};
use clippy_utils::ty::is_uninit_value_valid_for_ty;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::UNINIT_ASSUMED_INIT;

/// lint for `MaybeUninit::uninit().assume_init()` (we already have the latter)
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    if let hir::ExprKind::Call(callee, []) = recv.kind
        && callee.ty_rel_def(cx).is_diag_item(cx, sym::maybe_uninit_uninit)
        && !is_uninit_value_valid_for_ty(cx, cx.typeck_results().expr_ty_adjusted(expr))
    {
        span_lint(
            cx,
            UNINIT_ASSUMED_INIT,
            expr.span,
            "this call for this type may be undefined behavior",
        );
    }
}
