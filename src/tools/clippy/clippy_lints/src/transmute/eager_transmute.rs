use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{eq_expr_value, path_to_local, sym};
use rustc_abi::WrappingRange;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

use super::EAGER_TRANSMUTE;

fn peel_parent_unsafe_blocks<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    for (_, parent) in cx.tcx.hir_parent_iter(expr.hir_id) {
        match parent {
            Node::Block(_) => {},
            Node::Expr(e) if let ExprKind::Block(..) = e.kind => {},
            Node::Expr(e) => return Some(e),
            _ => break,
        }
    }
    None
}

fn range_fully_contained(from: WrappingRange, to: WrappingRange) -> bool {
    to.contains(from.start) && to.contains(from.end)
}

/// Checks if a given expression is a binary operation involving a local variable or is made up of
/// other (nested) binary expressions involving the local. There must be at least one local
/// reference that is the same as `local_expr`.
///
/// This is used as a heuristic to detect if a variable
/// is checked to be within the valid range of a transmuted type.
/// All of these would return true:
/// * `x < 4`
/// * `x < 4 && x > 1`
/// * `x.field < 4 && x.field > 1` (given `x.field`)
/// * `x.field < 4 && unrelated()`
/// * `(1..=3).contains(&x)`
fn binops_with_local(cx: &LateContext<'_>, local_expr: &Expr<'_>, expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Binary(_, lhs, rhs) => {
            binops_with_local(cx, local_expr, lhs) || binops_with_local(cx, local_expr, rhs)
        },
        ExprKind::MethodCall(path, receiver, [arg], _)
            if path.ident.name == sym::contains
                // ... `contains` called on some kind of range
                && let Some(receiver_adt) = cx.typeck_results().expr_ty(receiver).peel_refs().ty_adt_def()
                && let lang_items = cx.tcx.lang_items()
                && [
                    lang_items.range_from_struct(),
                    lang_items.range_inclusive_struct(),
                    lang_items.range_struct(),
                    lang_items.range_to_inclusive_struct(),
                    lang_items.range_to_struct()
                ].into_iter().any(|did| did == Some(receiver_adt.did())) =>
        {
            eq_expr_value(cx, local_expr, arg.peel_borrows())
        },
        _ => eq_expr_value(cx, local_expr, expr),
    }
}

/// Checks if an expression is a path to a local variable (with optional projections), e.g.
/// `x.field[0].field2` would return true.
fn is_local_with_projections(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Path(_) => path_to_local(expr).is_some(),
        ExprKind::Field(expr, _) | ExprKind::Index(expr, ..) => is_local_with_projections(expr),
        _ => false,
    }
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    transmutable: &'tcx Expr<'tcx>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
) -> bool {
    if let Some(then_some_call) = peel_parent_unsafe_blocks(cx, expr)
        && let ExprKind::MethodCall(path, receiver, [arg], _) = then_some_call.kind
        && cx.typeck_results().expr_ty(receiver).is_bool()
        && path.ident.name == sym::then_some
        && is_local_with_projections(transmutable)
        && binops_with_local(cx, transmutable, receiver)
        // we only want to lint if the target type has a niche that is larger than the one of the source type
        // e.g. `u8` to `NonZero<u8>` should lint, but `NonZero<u8>` to `u8` should not
        && let Ok(from_layout) = cx.tcx.layout_of(cx.typing_env().as_query_input(from_ty))
        && let Ok(to_layout) = cx.tcx.layout_of(cx.typing_env().as_query_input(to_ty))
        && match (from_layout.largest_niche, to_layout.largest_niche) {
            (Some(from_niche), Some(to_niche)) => !range_fully_contained(from_niche.valid_range, to_niche.valid_range),
            (None, Some(_)) => true,
            (_, None) => false,
        }
    {
        span_lint_and_then(
            cx,
            EAGER_TRANSMUTE,
            expr.span,
            "this transmute is always evaluated eagerly, even if the condition is false",
            |diag| {
                diag.multipart_suggestion(
                    "consider using `bool::then` to only transmute if the condition holds",
                    vec![
                        (path.ident.span, "then".into()),
                        (arg.span.shrink_to_lo(), "|| ".into()),
                    ],
                    Applicability::MaybeIncorrect,
                );
            },
        );
        true
    } else {
        false
    }
}
