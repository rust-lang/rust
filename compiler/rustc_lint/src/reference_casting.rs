use rustc_ast::Mutability;
use rustc_hir::{Expr, ExprKind, UnOp};
use rustc_middle::ty::layout::{LayoutOf as _, TyAndLayout};
use rustc_middle::ty::{self};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::sym;

use crate::lints::InvalidReferenceCastingDiag;
use crate::utils::peel_casts;
use crate::{LateContext, LateLintPass, LintContext};

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
            let init = cx.expr_or_init(e);
            let orig_cast = if init.span != e.span { Some(init.span) } else { None };

            // small cache to avoid recomputing needlessly computing peel_casts of init
            let mut peel_casts = {
                let mut peel_casts_cache = None;
                move || *peel_casts_cache.get_or_insert_with(|| peel_casts(cx, init))
            };

            if matches!(pat, PatternKind::Borrow { mutbl: Mutability::Mut } | PatternKind::Assign)
                && let Some(ty_has_interior_mutability) =
                    is_cast_from_ref_to_mut_ptr(cx, init, &mut peel_casts)
            {
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

            if let Some((from_ty_layout, to_ty_layout, e_alloc)) =
                is_cast_to_bigger_memory_layout(cx, init, &mut peel_casts)
            {
                cx.emit_span_lint(
                    INVALID_REFERENCE_CASTING,
                    expr.span,
                    InvalidReferenceCastingDiag::BiggerLayout {
                        orig_cast,
                        alloc: e_alloc.span,
                        from_ty: from_ty_layout.ty,
                        from_size: from_ty_layout.layout.size().bytes(),
                        to_ty: to_ty_layout.ty,
                        to_size: to_ty_layout.layout.size().bytes(),
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
    mut peel_casts: impl FnMut() -> (&'tcx Expr<'tcx>, bool),
) -> Option<bool> {
    let end_ty = cx.typeck_results().node_type(orig_expr.hir_id);

    // Bail out early if the end type is **not** a mutable pointer.
    if !matches!(end_ty.kind(), ty::RawPtr(_, Mutability::Mut)) {
        return None;
    }

    let (e, need_check_freeze) = peel_casts();

    let start_ty = cx.typeck_results().node_type(e.hir_id);
    if let ty::Ref(_, inner_ty, Mutability::Not) = start_ty.kind() {
        // If an UnsafeCell method is involved, we need to additionally check the
        // inner type for the presence of the Freeze trait (ie does NOT contain
        // an UnsafeCell), since in that case we would incorrectly lint on valid casts.
        //
        // Except on the presence of non concrete skeleton types (ie generics)
        // since there is no way to make it safe for arbitrary types.
        let inner_ty_has_interior_mutability =
            !inner_ty.is_freeze(cx.tcx, cx.typing_env()) && inner_ty.has_concrete_skeleton();
        (!need_check_freeze || !inner_ty_has_interior_mutability)
            .then_some(inner_ty_has_interior_mutability)
    } else {
        None
    }
}

fn is_cast_to_bigger_memory_layout<'tcx>(
    cx: &LateContext<'tcx>,
    orig_expr: &'tcx Expr<'tcx>,
    mut peel_casts: impl FnMut() -> (&'tcx Expr<'tcx>, bool),
) -> Option<(TyAndLayout<'tcx>, TyAndLayout<'tcx>, Expr<'tcx>)> {
    let end_ty = cx.typeck_results().node_type(orig_expr.hir_id);

    let ty::RawPtr(inner_end_ty, _) = end_ty.kind() else {
        return None;
    };

    let (e, _) = peel_casts();
    let start_ty = cx.typeck_results().node_type(e.hir_id);

    let ty::Ref(_, inner_start_ty, _) = start_ty.kind() else {
        return None;
    };

    // try to find the underlying allocation
    let e_alloc = cx.expr_or_init(e);
    let e_alloc =
        if let ExprKind::AddrOf(_, _, inner_expr) = e_alloc.kind { inner_expr } else { e_alloc };

    // if the current expr looks like this `&mut expr[index]` then just looking
    // at `expr[index]` won't give us the underlying allocation, so we just skip it
    // the same logic applies field access `&mut expr.field` and reborrows `&mut *expr`.
    if let ExprKind::Index(..) | ExprKind::Field(..) | ExprKind::Unary(UnOp::Deref, ..) =
        e_alloc.kind
    {
        return None;
    }

    let alloc_ty = cx.typeck_results().node_type(e_alloc.hir_id);

    // if we do not find it we bail out, as this may not be UB
    // see https://github.com/rust-lang/unsafe-code-guidelines/issues/256
    if alloc_ty.is_any_ptr() {
        return None;
    }

    let from_layout = cx.layout_of(*inner_start_ty).ok()?;

    // if the type isn't sized, we bail out, instead of potentially giving
    // the user a meaningless warning.
    if from_layout.is_unsized() {
        return None;
    }

    let alloc_layout = cx.layout_of(alloc_ty).ok()?;
    let to_layout = cx.layout_of(*inner_end_ty).ok()?;

    if to_layout.layout.size() > from_layout.layout.size()
        && to_layout.layout.size() > alloc_layout.layout.size()
    {
        Some((from_layout, to_layout, *e_alloc))
    } else {
        None
    }
}
