use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sym;
use clippy_utils::ty::get_type_diagnostic_name;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::{Span, Symbol};

use super::ITER_NTH;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
    iter_recv: &'tcx hir::Expr<'tcx>,
    iter_method: Symbol,
    iter_span: Span,
    nth_span: Span,
) -> bool {
    let caller_type = match get_type_diagnostic_name(cx, cx.typeck_results().expr_ty(iter_recv).peel_refs()) {
        Some(sym::Vec) => "`Vec`",
        Some(sym::VecDeque) => "`VecDeque`",
        _ if cx.typeck_results().expr_ty_adjusted(iter_recv).peel_refs().is_slice() => "slice",
        // caller is not a type that we want to lint
        _ => return false,
    };

    span_lint_and_then(
        cx,
        ITER_NTH,
        expr.span,
        format!("called `.{iter_method}().nth()` on a {caller_type}"),
        |diag| {
            let get_method = if iter_method == sym::iter_mut { "get_mut" } else { "get" };
            diag.span_suggestion_verbose(
                iter_span.to(nth_span),
                format!("`{get_method}` is equivalent but more concise"),
                get_method,
                Applicability::MachineApplicable,
            );
        },
    );

    true
}
