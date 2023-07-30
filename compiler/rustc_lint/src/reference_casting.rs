use rustc_ast::Mutability;
use rustc_hir::{Expr, ExprKind, MutTy, TyKind, UnOp};
use rustc_middle::ty;
use rustc_span::sym;

use crate::{lints::InvalidReferenceCastingDiag, LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `invalid_reference_casting` lint checks for casts of `&T` to `&mut T`
    /// without using interior mutability.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// # #![deny(invalid_reference_casting)]
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
    INVALID_REFERENCE_CASTING,
    Allow,
    "casts of `&T` to `&mut T` without interior mutability"
}

declare_lint_pass!(InvalidReferenceCasting => [INVALID_REFERENCE_CASTING]);

impl<'tcx> LateLintPass<'tcx> for InvalidReferenceCasting {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let ExprKind::Unary(UnOp::Deref, e) = &expr.kind else {
            return;
        };

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
            cx.emit_spanned_lint(INVALID_REFERENCE_CASTING, expr.span, InvalidReferenceCastingDiag);
        }
    }
}
