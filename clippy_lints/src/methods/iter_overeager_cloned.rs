use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::sym;

use super::ITER_OVEREAGER_CLONED;
use crate::redundant_clone::REDUNDANT_CLONE;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    cloned_call: &'tcx Expr<'_>,
    cloned_recv: &'tcx Expr<'_>,
    is_count: bool,
    needs_into_iter: bool,
) {
    let typeck = cx.typeck_results();
    if let Some(iter_id) = cx.tcx.get_diagnostic_item(sym::Iterator)
        && let Some(method_id) = typeck.type_dependent_def_id(expr.hir_id)
        && cx.tcx.trait_of_item(method_id) == Some(iter_id)
        && let Some(method_id) = typeck.type_dependent_def_id(cloned_call.hir_id)
        && cx.tcx.trait_of_item(method_id) == Some(iter_id)
        && let cloned_recv_ty = typeck.expr_ty_adjusted(cloned_recv)
        && let Some(iter_assoc_ty) = cx.get_associated_type(cloned_recv_ty, iter_id, "Item")
        && matches!(*iter_assoc_ty.kind(), ty::Ref(_, ty, _) if !is_copy(cx, ty))
    {
        if needs_into_iter
            && let Some(into_iter_id) = cx.tcx.get_diagnostic_item(sym::IntoIterator)
            && !implements_trait(cx, iter_assoc_ty, into_iter_id, &[])
        {
            return;
        }

        let (lint, msg, trailing_clone) = if is_count {
            (REDUNDANT_CLONE, "unneeded cloning of iterator items", "")
        } else {
            (ITER_OVEREAGER_CLONED, "unnecessarily eager cloning of iterator items", ".cloned()")
        };

        span_lint_and_then(
            cx,
            lint,
            expr.span,
            msg,
            |diag| {
                let method_span = expr.span.with_lo(cloned_call.span.hi());
                if let Some(mut snip) = snippet_opt(cx, method_span) {
                    snip.push_str(trailing_clone);
                    let replace_span = expr.span.with_lo(cloned_recv.span.hi());
                    diag.span_suggestion(replace_span, "try this", snip, Applicability::MachineApplicable);
                }
            }
        );
    }
}
