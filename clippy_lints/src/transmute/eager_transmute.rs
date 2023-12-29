use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_normalizable;
use clippy_utils::{path_to_local, path_to_local_id};
use rustc_abi::WrappingRange;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

use super::EAGER_TRANSMUTE;

fn peel_parent_unsafe_blocks<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    for (_, parent) in cx.tcx.hir().parent_iter(expr.hir_id) {
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
        && path.ident.name == sym!(then_some)
        && let ExprKind::Binary(_, lhs, rhs) = receiver.kind
        && let Some(local_id) = path_to_local(transmutable)
        && (path_to_local_id(lhs, local_id) || path_to_local_id(rhs, local_id))
        && is_normalizable(cx, cx.param_env, from_ty)
        && is_normalizable(cx, cx.param_env, to_ty)
        // we only want to lint if the target type has a niche that is larger than the one of the source type
        // e.g. `u8` to `NonZeroU8` should lint, but `NonZeroU8` to `u8` should not
        && let Ok(from_layout) = cx.tcx.layout_of(cx.param_env.and(from_ty))
        && let Ok(to_layout) = cx.tcx.layout_of(cx.param_env.and(to_ty))
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
