use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::peel_blocks;
use clippy_utils::res::{MaybeDef, MaybeResPath};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::visitors::for_each_expr_without_closures;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, ExprKind, HirId, PatKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use std::ops::ControlFlow;

use super::MANUAL_OPTION_ZIP;

/// Checks for `a.and_then(|a| b.map(|b| (a, b)))` and suggests `a.zip(b)`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    recv: &'tcx hir::Expr<'_>,
    arg: &'tcx hir::Expr<'_>,
    msrv: Msrv,
) {
    // Looking for: `a.and_then(|a| b.map(|b| (a, b)))`.
    // `and_then(|a| ...)`
    if let ExprKind::Closure(&hir::Closure { body: outer_body_id, .. }) = arg.kind
        && let hir::Body { params: [outer_param], value: outer_value, .. } = cx.tcx.hir_body(outer_body_id)
        && let PatKind::Binding(_, outer_param_id, _, None) = outer_param.pat.kind
        && cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Option)
        // `b.map(|b| ...)`
        && let ExprKind::MethodCall(method_path, map_recv, [map_arg], _) = peel_blocks(outer_value).kind
        && method_path.ident.name == sym::map
        && cx.typeck_results().expr_ty(map_recv).is_diag_item(cx, sym::Option)
        // `b` does not reference the outer closure parameter `a`.
        && for_each_expr_without_closures(map_recv, |e| {
            if e.res_local_id() == Some(outer_param_id) {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        }).is_none()
        // `|b| (a, b)`
        && let ExprKind::Closure(&hir::Closure { body: inner_body_id, .. }) = map_arg.kind
        && let hir::Body { params: [inner_param], value: inner_value, .. } = cx.tcx.hir_body(inner_body_id)
        && let PatKind::Binding(_, inner_param_id, _, None) = inner_param.pat.kind
        // `(a, b)` or `(b, a)` — tuple of outer and inner param in either order.
        && let ExprKind::Tup([first, second]) = peel_blocks(inner_value).kind
        && let Some((zip_recv, zip_arg)) = zip_operands(first, second, outer_param_id, inner_param_id, recv, map_recv)
        // `Option.zip()` is available.
        && msrv.meets(cx, msrvs::OPTION_ZIP)
    {
        let mut applicability = Applicability::MachineApplicable;
        let zip_recv_snip = snippet_with_applicability(cx, zip_recv.span, "_", &mut applicability);
        let zip_arg_snip = snippet_with_applicability(cx, zip_arg.span, "_", &mut applicability);
        let suggestion = format!("{zip_recv_snip}.zip({zip_arg_snip})");

        span_lint_and_sugg(
            cx,
            MANUAL_OPTION_ZIP,
            expr.span,
            "manual implementation of `Option::zip`",
            "use",
            suggestion,
            applicability,
        );
    }
}

/// Given the two tuple elements and the `and_then` receiver / `map` receiver, returns the
/// `(zip_receiver, zip_argument)` expressions for the `.zip()` suggestion.
///
/// For `(outer, inner)` order the zip is `recv.zip(map_recv)`.
/// For `(inner, outer)` (reversed) the zip is `map_recv.zip(recv)`.
/// Returns `None` if the tuple elements don't match either order.
fn zip_operands<'a>(
    first: &hir::Expr<'_>,
    second: &hir::Expr<'_>,
    outer_param_id: HirId,
    inner_param_id: HirId,
    recv: &'a hir::Expr<'a>,
    map_recv: &'a hir::Expr<'a>,
) -> Option<(&'a hir::Expr<'a>, &'a hir::Expr<'a>)> {
    if first.res_local_id() == Some(outer_param_id) && second.res_local_id() == Some(inner_param_id) {
        Some((recv, map_recv))
    } else if first.res_local_id() == Some(inner_param_id) && second.res_local_id() == Some(outer_param_id) {
        Some((map_recv, recv))
    } else {
        None
    }
}
