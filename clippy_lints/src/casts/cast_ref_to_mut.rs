use clippy_utils::diagnostics::span_lint;
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, MutTy, Mutability, TyKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty;

use super::CAST_REF_TO_MUT;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if_chain! {
        if let ExprKind::Unary(UnOp::Deref, e) = &expr.kind;
        if let ExprKind::Cast(e, t) = &e.kind;
        if let TyKind::Ptr(MutTy { mutbl: Mutability::Mut, .. }) = t.kind;
        if let ExprKind::Cast(e, t) = &e.kind;
        if let TyKind::Ptr(MutTy { mutbl: Mutability::Not, .. }) = t.kind;
        if let ty::Ref(..) = cx.typeck_results().node_type(e.hir_id).kind();
        then {
            span_lint(
                cx,
                CAST_REF_TO_MUT,
                expr.span,
                "casting `&T` to `&mut T` may cause undefined behavior, consider instead using an `UnsafeCell`",
            );
        }
    }
}
