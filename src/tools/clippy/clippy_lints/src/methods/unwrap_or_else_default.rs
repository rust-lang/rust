//! Lint for `some_result_or_option.unwrap_or_else(Default::default)`

use super::UNWRAP_OR_ELSE_DEFAULT;
use clippy_utils::{
    diagnostics::span_lint_and_sugg, is_default_equivalent_call, source::snippet_with_applicability,
    ty::is_type_diagnostic_item,
};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    u_arg: &'tcx hir::Expr<'_>,
) {
    // something.unwrap_or_else(Default::default)
    // ^^^^^^^^^- recv          ^^^^^^^^^^^^^^^^- u_arg
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^- expr
    let recv_ty = cx.typeck_results().expr_ty(recv);
    let is_option = is_type_diagnostic_item(cx, recv_ty, sym::Option);
    let is_result = is_type_diagnostic_item(cx, recv_ty, sym::Result);

    if_chain! {
        if is_option || is_result;
        if is_default_equivalent_call(cx, u_arg);
        then {
            let mut applicability = Applicability::MachineApplicable;

            span_lint_and_sugg(
                cx,
                UNWRAP_OR_ELSE_DEFAULT,
                expr.span,
                "use of `.unwrap_or_else(..)` to construct default value",
                "try",
                format!(
                    "{}.unwrap_or_default()",
                    snippet_with_applicability(cx, recv.span, "..", &mut applicability)
                ),
                applicability,
            );
        }
    }
}
