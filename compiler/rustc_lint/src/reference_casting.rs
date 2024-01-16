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
        if let Some((e, pat)) = borrow_or_assign(cx, expr) {
            if matches!(pat, PatternKind::Borrow { mutbl: Mutability::Mut } | PatternKind::Assign) {
                let init = cx.expr_or_init(e);

                let Some(ty_has_interior_mutability) = is_cast_from_ref_to_mut_ptr(cx, init) else {
                    return;
                };
                let orig_cast = if init.span != e.span { Some(init.span) } else { None };
                let ty_has_interior_mutability = ty_has_interior_mutability.then_some(());

                cx.emit_span_lint(
                    INVALID_REFERENCE_CASTING,
                    expr.span,
                    if pat == PatternKind::Assign {
                        InvalidReferenceCastingDiag::AssignToRef {
                            orig_cast,
                            ty_has_interior_mutability,
                        }
                    } else {
                        InvalidReferenceCastingDiag::BorrowAsMut {
                            orig_cast,
                            ty_has_interior_mutability,
                        }
                    },
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternKind {
    Borrow { mutbl: Mutability },
    Assign,
}

fn borrow_or_assign<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, PatternKind)> {
    fn deref_assign_or_addr_of<'tcx>(
        expr: &'tcx Expr<'tcx>,
    ) -> Option<(&'tcx Expr<'tcx>, PatternKind)> {
        // &(mut) <expr>
        let (inner, pat) = if let ExprKind::AddrOf(_, mutbl, expr) = expr.kind {
            (expr, PatternKind::Borrow { mutbl })
        // <expr> = ...
        } else if let ExprKind::Assign(expr, _, _) = expr.kind {
            (expr, PatternKind::Assign)
        // <expr> += ...
        } else if let ExprKind::AssignOp(_, expr, _) = expr.kind {
            (expr, PatternKind::Assign)
        } else {
            return None;
        };

        // *<inner>
        let ExprKind::Unary(UnOp::Deref, e) = &inner.kind else {
            return None;
        };
        Some((e, pat))
    }

    fn ptr_write<'tcx>(
        cx: &LateContext<'tcx>,
        e: &'tcx Expr<'tcx>,
    ) -> Option<(&'tcx Expr<'tcx>, PatternKind)> {
        if let ExprKind::Call(path, [arg_ptr, _arg_val]) = e.kind
            && let ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && matches!(
                cx.tcx.get_diagnostic_name(def_id),
                Some(sym::ptr_write | sym::ptr_write_volatile | sym::ptr_write_unaligned)
            )
        {
            Some((arg_ptr, PatternKind::Assign))
        } else {
            None
        }
    }

    deref_assign_or_addr_of(e).or_else(|| ptr_write(cx, e))
}

fn is_cast_from_ref_to_mut_ptr<'tcx>(
    cx: &LateContext<'tcx>,
    orig_expr: &'tcx Expr<'tcx>,
) -> Option<bool> {
    let end_ty = cx.typeck_results().node_type(orig_expr.hir_id);

    // Bail out early if the end type is **not** a mutable pointer.
    if !matches!(end_ty.kind(), ty::RawPtr(TypeAndMut { ty: _, mutbl: Mutability::Mut })) {
        return None;
    }

    let (e, need_check_freeze) = peel_casts(cx, orig_expr);

    let start_ty = cx.typeck_results().node_type(e.hir_id);
    if let ty::Ref(_, inner_ty, Mutability::Not) = start_ty.kind() {
        // If an UnsafeCell method is involved, we need to additionally check the
        // inner type for the presence of the Freeze trait (ie does NOT contain
        // an UnsafeCell), since in that case we would incorrectly lint on valid casts.
        //
        // Except on the presence of non concrete skeleton types (ie generics)
        // since there is no way to make it safe for arbitrary types.
        let inner_ty_has_interior_mutability =
            !inner_ty.is_freeze(cx.tcx, cx.param_env) && inner_ty.has_concrete_skeleton();
        (!need_check_freeze || !inner_ty_has_interior_mutability)
            .then_some(inner_ty_has_interior_mutability)
    } else {
        None
    }
}

fn peel_casts<'tcx>(cx: &LateContext<'tcx>, mut e: &'tcx Expr<'tcx>) -> (&'tcx Expr<'tcx>, bool) {
    let mut gone_trough_unsafe_cell_raw_get = false;

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
                gone_trough_unsafe_cell_raw_get = true;
            }
            arg
        } else {
            let init = cx.expr_or_init(e);
            if init.hir_id != e.hir_id {
                init
            } else {
                break;
            }
        };
    }

    (e, gone_trough_unsafe_cell_raw_get)
}
