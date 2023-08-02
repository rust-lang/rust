use rustc_ast::Mutability;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{def::Res, Expr, ExprKind, HirId, Local, QPath, StmtKind, UnOp};
use rustc_middle::ty::{self, TypeAndMut};
use rustc_span::{sym, Span};

use crate::{lints::InvalidReferenceCastingDiag, LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `invalid_reference_casting` lint checks for casts of `&T` to `&mut T`
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
    INVALID_REFERENCE_CASTING,
    Deny,
    "casts of `&T` to `&mut T` without interior mutability"
}

#[derive(Default)]
pub struct InvalidReferenceCasting {
    casted: FxHashMap<HirId, Span>,
}

impl_lint_pass!(InvalidReferenceCasting => [INVALID_REFERENCE_CASTING]);

impl<'tcx> LateLintPass<'tcx> for InvalidReferenceCasting {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx rustc_hir::Stmt<'tcx>) {
        let StmtKind::Local(local) = stmt.kind else {
            return;
        };
        let Local { init: Some(init), els: None, .. } = local else {
            return;
        };

        if is_cast_from_const_to_mut(cx, init) {
            self.casted.insert(local.pat.hir_id, init.span);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        // &mut <expr>
        let inner = if let ExprKind::AddrOf(_, Mutability::Mut, expr) = expr.kind {
            expr
        // <expr> = ...
        } else if let ExprKind::Assign(expr, _, _) = expr.kind {
            expr
        // <expr> += ...
        } else if let ExprKind::AssignOp(_, expr, _) = expr.kind {
            expr
        } else {
            return;
        };

        let ExprKind::Unary(UnOp::Deref, e) = &inner.kind else {
            return;
        };

        let orig_cast = if is_cast_from_const_to_mut(cx, e) {
            None
        } else if let ExprKind::Path(QPath::Resolved(_, path)) = e.kind
            && let Res::Local(hir_id) = &path.res
            && let Some(orig_cast) = self.casted.get(hir_id) {
            Some(*orig_cast)
        } else {
            return;
        };

        cx.emit_spanned_lint(
            INVALID_REFERENCE_CASTING,
            expr.span,
            if matches!(expr.kind, ExprKind::AddrOf(..)) {
                InvalidReferenceCastingDiag::BorrowAsMut { orig_cast }
            } else {
                InvalidReferenceCastingDiag::AssignToRef { orig_cast }
            },
        );
    }
}

fn is_cast_from_const_to_mut<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) -> bool {
    let e = e.peel_blocks();

    // <expr> as *mut ...
    let e = if let ExprKind::Cast(e, t) = e.kind
        && let ty::RawPtr(TypeAndMut { mutbl: Mutability::Mut, .. }) = cx.typeck_results().node_type(t.hir_id).kind() {
        e
    // <expr>.cast_mut()
    } else if let ExprKind::MethodCall(_, expr, [], _) = e.kind
        && let Some(def_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
        && cx.tcx.is_diagnostic_item(sym::ptr_cast_mut, def_id) {
        expr
    } else {
        return false;
    };

    let e = e.peel_blocks();

    // <expr> as *const ...
    let e = if let ExprKind::Cast(e, t) = e.kind
        && let ty::RawPtr(TypeAndMut { mutbl: Mutability::Not, .. }) = cx.typeck_results().node_type(t.hir_id).kind() {
        e
    // ptr::from_ref(<expr>)
    } else if let ExprKind::Call(path, [arg]) = e.kind
        && let ExprKind::Path(ref qpath) = path.kind
        && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
        && cx.tcx.is_diagnostic_item(sym::ptr_from_ref, def_id) {
        arg
    } else {
        return false;
    };

    let e = e.peel_blocks();
    matches!(cx.typeck_results().node_type(e.hir_id).kind(), ty::Ref(_, _, Mutability::Not))
}
