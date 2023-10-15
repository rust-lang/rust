use rustc_ast::Mutability;
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_middle::ty::{self, TypeAndMut};
use rustc_span::sym;

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

declare_lint_pass!(InvalidReferenceCasting => [INVALID_REFERENCE_CASTING]);

impl<'tcx> LateLintPass<'tcx> for InvalidReferenceCasting {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let Some((is_assignment, e)) = is_operation_we_care_about(cx, expr) else {
            return;
        };

        let init = cx.expr_or_init(e);

        let Some(ty_has_interior_mutability) = is_cast_from_const_to_mut(cx, init) else {
            return;
        };
        let orig_cast = if init.span != e.span { Some(init.span) } else { None };
        let ty_has_interior_mutability = ty_has_interior_mutability.then_some(());

        cx.emit_spanned_lint(
            INVALID_REFERENCE_CASTING,
            expr.span,
            if is_assignment {
                InvalidReferenceCastingDiag::AssignToRef { orig_cast, ty_has_interior_mutability }
            } else {
                InvalidReferenceCastingDiag::BorrowAsMut { orig_cast, ty_has_interior_mutability }
            },
        );
    }
}

fn is_operation_we_care_about<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'tcx>,
) -> Option<(bool, &'tcx Expr<'tcx>)> {
    fn deref_assign_or_addr_of<'tcx>(expr: &'tcx Expr<'tcx>) -> Option<(bool, &'tcx Expr<'tcx>)> {
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
            return None;
        };

        if let ExprKind::Unary(UnOp::Deref, e) = &inner.kind {
            Some((!matches!(expr.kind, ExprKind::AddrOf(..)), e))
        } else {
            None
        }
    }

    fn ptr_write<'tcx>(
        cx: &LateContext<'tcx>,
        e: &'tcx Expr<'tcx>,
    ) -> Option<(bool, &'tcx Expr<'tcx>)> {
        if let ExprKind::Call(path, [arg_ptr, _arg_val]) = e.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && matches!(
                cx.tcx.get_diagnostic_name(def_id),
                Some(sym::ptr_write | sym::ptr_write_volatile | sym::ptr_write_unaligned)
            )
        {
            Some((true, arg_ptr))
        } else {
            None
        }
    }

    deref_assign_or_addr_of(e).or_else(|| ptr_write(cx, e))
}

fn is_cast_from_const_to_mut<'tcx>(
    cx: &LateContext<'tcx>,
    orig_expr: &'tcx Expr<'tcx>,
) -> Option<bool> {
    let mut need_check_freeze = false;
    let mut e = orig_expr;

    let end_ty = cx.typeck_results().node_type(orig_expr.hir_id);

    // Bail out early if the end type is **not** a mutable pointer.
    if !matches!(end_ty.kind(), ty::RawPtr(TypeAndMut { ty: _, mutbl: Mutability::Mut })) {
        return None;
    }

    loop {
        e = e.peel_blocks();
        // <expr> as ...
        e = if let ExprKind::Cast(expr, _) = e.kind {
            expr
        // <expr>.cast(), <expr>.cast_mut() or <expr>.cast_const()
        } else if let ExprKind::MethodCall(_, expr, [], _) = e.kind
            && let Some(def_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
            && matches!(
                cx.tcx.get_diagnostic_name(def_id),
                Some(sym::ptr_cast | sym::const_ptr_cast | sym::ptr_cast_mut | sym::ptr_cast_const)
            )
        {
            expr
        // ptr::from_ref(<expr>), UnsafeCell::raw_get(<expr>) or mem::transmute<_, _>(<expr>)
        } else if let ExprKind::Call(path, [arg]) = e.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && matches!(
                cx.tcx.get_diagnostic_name(def_id),
                Some(sym::ptr_from_ref | sym::unsafe_cell_raw_get | sym::transmute)
            )
        {
            if cx.tcx.is_diagnostic_item(sym::unsafe_cell_raw_get, def_id) {
                need_check_freeze = true;
            }
            arg
        } else {
            break;
        };
    }

    let start_ty = cx.typeck_results().node_type(e.hir_id);
    if let ty::Ref(_, inner_ty, Mutability::Not) = start_ty.kind() {
        // If an UnsafeCell method is involved we need to additionaly check the
        // inner type for the presence of the Freeze trait (ie does NOT contain
        // an UnsafeCell), since in that case we would incorrectly lint on valid casts.
        //
        // We also consider non concrete skeleton types (ie generics)
        // to be an issue since there is no way to make it safe for abitrary types.
        let inner_ty_has_interior_mutability =
            !inner_ty.is_freeze(cx.tcx, cx.param_env) && inner_ty.has_concrete_skeleton();
        (!need_check_freeze || !inner_ty_has_interior_mutability)
            .then_some(inner_ty_has_interior_mutability)
    } else {
        None
    }
}
