use rustc_ast::Mutability;
use rustc_hir::{Expr, ExprKind, MutTy, TyKind, UnOp};
use rustc_middle::ty;
use rustc_span::sym;

use crate::{lints::CastRefToMutDiag, LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `cast_ref_to_mut` lint checks for casts of `&T` to `&mut T`
    /// without using interior mutability.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// fn x(r: &i32) {
    ///     unsafe {
    ///         *(r as *const i32 as *mut i32) += 1;
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Casting `&T` to `&mut T` without using interior mutability is undefined behavior,
    /// as it's a violation of Rust reference aliasing requirements.
    ///
    /// `UnsafeCell` is the only way to obtain aliasable data that is considered
    /// mutable.
    CAST_REF_TO_MUT,
    Deny,
    "casts of `&T` to `&mut T` without interior mutability"
}

declare_lint_pass!(CastRefToMut => [CAST_REF_TO_MUT]);

impl<'tcx> LateLintPass<'tcx> for CastRefToMut {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let ExprKind::Unary(UnOp::Deref, e) = &expr.kind else { return; };

        let e = e.peel_blocks();
        let e = if let ExprKind::Cast(e, t) = e.kind
            && let TyKind::Ptr(MutTy { mutbl: Mutability::Mut, .. }) = t.kind {
            e
        } else if let ExprKind::MethodCall(_, expr, [], _) = e.kind
            && let Some(def_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
            && cx.tcx.is_diagnostic_item(sym::ptr_cast_mut, def_id) {
            expr
        } else {
            return;
        };

        let e = e.peel_blocks();
        let e = if let ExprKind::Cast(e, t) = e.kind
            && let TyKind::Ptr(MutTy { mutbl: Mutability::Not, .. }) = t.kind {
            e
        } else if let ExprKind::Call(path, [arg]) = e.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::ptr_from_ref, def_id) {
            arg
        } else {
            return;
        };

        let e = e.peel_blocks();
        if let ty::Ref(..) = cx.typeck_results().node_type(e.hir_id).kind() {
            cx.emit_spanned_lint(CAST_REF_TO_MUT, expr.span, CastRefToMutDiag);
        }
    }
}
