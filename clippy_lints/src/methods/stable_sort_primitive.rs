use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_slice_of_primitives;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;

use super::STABLE_SORT_PRIMITIVE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, recv: &'tcx Expr<'_>) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(e.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_method(method_id)
        && cx.tcx.type_of(impl_id).is_slice()
        && let Some(slice_type) = is_slice_of_primitives(cx, recv)
    {
        span_lint_and_then(
            cx,
            STABLE_SORT_PRIMITIVE,
            e.span,
            &format!("used `sort` on primitive type `{}`", slice_type),
            |diag| {
                let mut app = Applicability::MachineApplicable;
                let recv_snip = snippet_with_context(cx, recv.span, e.span.ctxt(), "..", &mut app).0;
                diag.span_suggestion(e.span, "try", format!("{}.sort_unstable()", recv_snip), app);
                diag.note(
                    "an unstable sort typically performs faster without any observable difference for this data type",
                );
            },
        );
    }
}
