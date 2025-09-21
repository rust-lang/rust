use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::InteriorMut;
use clippy_utils::{SpanlessEq, eq_expr_value, find_binding_init, hash_expr, path_to_local, search_same};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;

use super::IFS_SAME_COND;

fn method_caller_is_mutable<'tcx>(
    cx: &LateContext<'tcx>,
    caller_expr: &Expr<'_>,
    interior_mut: &mut InteriorMut<'tcx>,
) -> bool {
    let caller_ty = cx.typeck_results().expr_ty(caller_expr);

    interior_mut.is_interior_mut_ty(cx, caller_ty)
        || caller_ty.is_mutable_ptr()
        // `find_binding_init` will return the binding iff its not mutable
        || path_to_local(caller_expr)
            .and_then(|hid| find_binding_init(cx, hid))
            .is_none()
}

/// Implementation of `IFS_SAME_COND`.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, conds: &[&Expr<'_>], interior_mut: &mut InteriorMut<'tcx>) {
    for group in search_same(
        conds,
        |e| hash_expr(cx, e),
        |lhs, rhs| {
            // Ignore eq_expr side effects iff one of the expression kind is a method call
            // and the caller is not a mutable, including inner mutable type.
            if let ExprKind::MethodCall(_, caller, _, _) = lhs.kind {
                if method_caller_is_mutable(cx, caller, interior_mut) {
                    false
                } else {
                    SpanlessEq::new(cx).eq_expr(lhs, rhs)
                }
            } else {
                eq_expr_value(cx, lhs, rhs)
            }
        },
    ) {
        let spans: Vec<_> = group.into_iter().map(|expr| expr.span).collect();
        span_lint(cx, IFS_SAME_COND, spans, "these `if` branches have the same condition");
    }
}
