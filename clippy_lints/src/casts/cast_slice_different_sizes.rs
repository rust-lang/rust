use clippy_utils::{diagnostics::span_lint_and_then, meets_msrv, msrvs, source::snippet_opt};
use if_chain::if_chain;
use rustc_ast::Mutability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, layout::LayoutOf, Ty, TypeAndMut};
use rustc_semver::RustcVersion;

use super::CAST_SLICE_DIFFERENT_SIZES;

fn is_child_of_cast(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let map = cx.tcx.hir();
    if_chain! {
        if let Some(parent_id) = map.find_parent_node(expr.hir_id);
        if let Some(parent) = map.find(parent_id);
        then {
            let expr = match parent {
                Node::Block(block) => {
                    if let Some(parent_expr) = block.expr {
                        parent_expr
                    } else {
                        return false;
                    }
                },
                Node::Expr(expr) => expr,
                _ => return false,
            };

            matches!(expr.kind, ExprKind::Cast(..))
        } else {
            false
        }
    }
}

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, msrv: &Option<RustcVersion>) {
    // suggestion is invalid if `ptr::slice_from_raw_parts` does not exist
    if !meets_msrv(msrv.as_ref(), &msrvs::PTR_SLICE_RAW_PARTS) {
        return;
    }

    // if this cast is the child of another cast expression then don't emit something for it, the full
    // chain will be analyzed
    if is_child_of_cast(cx, expr) {
        return;
    }

    if let Some((from_slice_ty, to_slice_ty)) = expr_cast_chain_tys(cx, expr) {
        if let (Ok(from_layout), Ok(to_layout)) = (cx.layout_of(from_slice_ty.ty), cx.layout_of(to_slice_ty.ty)) {
            let from_size = from_layout.size.bytes();
            let to_size = to_layout.size.bytes();
            if from_size != to_size && from_size != 0 && to_size != 0 {
                span_lint_and_then(
                    cx,
                    CAST_SLICE_DIFFERENT_SIZES,
                    expr.span,
                    &format!(
                        "casting between raw pointers to `[{}]` (element size {}) and `[{}]` (element size {}) does not adjust the count",
                        from_slice_ty, from_size, to_slice_ty, to_size,
                    ),
                    |diag| {
                        let cast_expr = match expr.kind {
                            ExprKind::Cast(cast_expr, ..) => cast_expr,
                            _ => unreachable!("expr should be a cast as checked by expr_cast_chain_tys"),
                        };
                        let ptr_snippet = snippet_opt(cx, cast_expr.span).unwrap();

                        let (mutbl_fn_str, mutbl_ptr_str) = match to_slice_ty.mutbl {
                            Mutability::Mut => ("_mut", "mut"),
                            Mutability::Not => ("", "const"),
                        };
                        let sugg = format!(
                            "core::ptr::slice_from_raw_parts{mutbl_fn_str}({ptr_snippet} as *{mutbl_ptr_str} {to_slice_ty}, ..)"
                        );

                        diag.span_suggestion(
                            expr.span,
                            &format!("replace with `ptr::slice_from_raw_parts{mutbl_fn_str}`"),
                            sugg,
                            rustc_errors::Applicability::HasPlaceholders,
                        );
                    },
                );
            }
        }
    }
}

/// Returns the type T of the pointed to *const [T] or *mut [T] and the mutability of the slice if
/// the type is one of those slices
fn get_raw_slice_ty_mut(ty: Ty<'_>) -> Option<TypeAndMut<'_>> {
    match ty.kind() {
        ty::RawPtr(TypeAndMut { ty: slice_ty, mutbl }) => match slice_ty.kind() {
            ty::Slice(ty) => Some(TypeAndMut { ty: *ty, mutbl: *mutbl }),
            _ => None,
        },
        _ => None,
    }
}

/// Returns the pair (original ptr T, final ptr U) if the expression is composed of casts
/// Returns None if the expr is not a Cast
fn expr_cast_chain_tys<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>) -> Option<(TypeAndMut<'tcx>, TypeAndMut<'tcx>)> {
    if let ExprKind::Cast(cast_expr, _cast_to_hir_ty) = expr.peel_blocks().kind {
        let cast_to = cx.typeck_results().expr_ty(expr);
        let to_slice_ty = get_raw_slice_ty_mut(cast_to)?;
        if let Some((inner_from_ty, _inner_to_ty)) = expr_cast_chain_tys(cx, cast_expr) {
            Some((inner_from_ty, to_slice_ty))
        } else {
            let cast_from = cx.typeck_results().expr_ty(cast_expr);
            let from_slice_ty = get_raw_slice_ty_mut(cast_from)?;
            Some((from_slice_ty, to_slice_ty))
        }
    } else {
        None
    }
}
