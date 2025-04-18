use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source;
use rustc_ast::Mutability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, Ty, TypeAndMut};

use super::CAST_SLICE_DIFFERENT_SIZES;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>, msrv: Msrv) {
    // if this cast is the child of another cast expression then don't emit something for it, the full
    // chain will be analyzed
    if is_child_of_cast(cx, expr) {
        return;
    }

    if let Some(CastChainInfo {
        left_cast,
        start_ty,
        end_ty,
    }) = expr_cast_chain_tys(cx, expr)
        && let (Ok(from_layout), Ok(to_layout)) = (cx.layout_of(start_ty.ty), cx.layout_of(end_ty.ty))
    {
        let from_size = from_layout.size.bytes();
        let to_size = to_layout.size.bytes();
        if from_size != to_size && from_size != 0 && to_size != 0 && msrv.meets(cx, msrvs::PTR_SLICE_RAW_PARTS) {
            span_lint_and_then(
                cx,
                CAST_SLICE_DIFFERENT_SIZES,
                expr.span,
                format!(
                    "casting between raw pointers to `[{}]` (element size {from_size}) and `[{}]` (element size {to_size}) does not adjust the count",
                    start_ty.ty, end_ty.ty,
                ),
                |diag| {
                    let ptr_snippet = source::snippet(cx, left_cast.span, "..");

                    let (mutbl_fn_str, mutbl_ptr_str) = match end_ty.mutbl {
                        Mutability::Mut => ("_mut", "mut"),
                        Mutability::Not => ("", "const"),
                    };
                    let sugg = format!(
                        "core::ptr::slice_from_raw_parts{mutbl_fn_str}({ptr_snippet} as *{mutbl_ptr_str} {}, ..)",
                        // get just the ty from the TypeAndMut so that the printed type isn't something like `mut
                        // T`, extract just the `T`
                        end_ty.ty
                    );

                    diag.span_suggestion(
                        expr.span,
                        format!("replace with `ptr::slice_from_raw_parts{mutbl_fn_str}`"),
                        sugg,
                        rustc_errors::Applicability::HasPlaceholders,
                    );
                },
            );
        }
    }
}

fn is_child_of_cast(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let parent = cx.tcx.parent_hir_node(expr.hir_id);
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
}

/// Returns the type T of the pointed to *const [T] or *mut [T] and the mutability of the slice if
/// the type is one of those slices
fn get_raw_slice_ty_mut(ty: Ty<'_>) -> Option<TypeAndMut<'_>> {
    match ty.kind() {
        ty::RawPtr(slice_ty, mutbl) => match slice_ty.kind() {
            ty::Slice(ty) => Some(TypeAndMut { ty: *ty, mutbl: *mutbl }),
            _ => None,
        },
        _ => None,
    }
}

struct CastChainInfo<'tcx> {
    /// The left most part of the cast chain, or in other words, the first cast in the chain
    /// Used for diagnostics
    left_cast: &'tcx Expr<'tcx>,
    /// The starting type of the cast chain
    start_ty: TypeAndMut<'tcx>,
    /// The final type of the cast chain
    end_ty: TypeAndMut<'tcx>,
}

/// Returns a `CastChainInfo` with the left-most cast in the chain and the original ptr T and final
/// ptr U if the expression is composed of casts.
/// Returns None if the expr is not a Cast
fn expr_cast_chain_tys<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<CastChainInfo<'tcx>> {
    if let ExprKind::Cast(cast_expr, _cast_to_hir_ty) = expr.peel_blocks().kind {
        let cast_to = cx.typeck_results().expr_ty(expr);
        let to_slice_ty = get_raw_slice_ty_mut(cast_to)?;

        // If the expression that makes up the source of this cast is itself a cast, recursively
        // call `expr_cast_chain_tys` and update the end type with the final target type.
        // Otherwise, this cast is not immediately nested, just construct the info for this cast
        if let Some(prev_info) = expr_cast_chain_tys(cx, cast_expr) {
            Some(CastChainInfo {
                end_ty: to_slice_ty,
                ..prev_info
            })
        } else {
            let cast_from = cx.typeck_results().expr_ty(cast_expr);
            let from_slice_ty = get_raw_slice_ty_mut(cast_from)?;
            Some(CastChainInfo {
                left_cast: cast_expr,
                start_ty: from_slice_ty,
                end_ty: to_slice_ty,
            })
        }
    } else {
        None
    }
}
