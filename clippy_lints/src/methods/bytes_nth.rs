use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::BYTES_NTH;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, recv: &'tcx Expr<'tcx>, n_arg: &'tcx Expr<'tcx>) {
    if_chain! {
        let ty = cx.typeck_results().expr_ty(recv).peel_refs();
        let caller_type = if is_type_diagnostic_item(cx, ty, sym::string_type) {
            Some("String")
        } else if ty.is_str() {
            Some("str")
        } else {
            None
        };
        if let Some(caller_type) = caller_type;
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                BYTES_NTH,
                expr.span,
                &format!("called `.byte().nth()` on a `{}`", caller_type),
                "try",
                format!(
                    "{}.as_bytes().get({})",
                    snippet_with_applicability(cx, recv.span, "..", &mut applicability),
                    snippet_with_applicability(cx, n_arg.span, "..", &mut applicability)
                ),
                applicability,
            );
        }
    }
}
