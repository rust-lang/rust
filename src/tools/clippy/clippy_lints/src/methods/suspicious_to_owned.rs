use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_diag_trait_item;
use clippy_utils::source::snippet_with_context;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::sym;

use super::SUSPICIOUS_TO_OWNED;

pub fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) -> bool {
    if_chain! {
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if is_diag_trait_item(cx, method_def_id, sym::ToOwned);
        let input_type = cx.typeck_results().expr_ty(expr);
        if let ty::Adt(adt, _) = cx.typeck_results().expr_ty(expr).kind();
        if cx.tcx.is_diagnostic_item(sym::Cow, adt.did());
        then {
            let mut app = Applicability::MaybeIncorrect;
            let recv_snip = snippet_with_context(cx, recv.span, expr.span.ctxt(), "..", &mut app).0;
            span_lint_and_sugg(
                cx,
                SUSPICIOUS_TO_OWNED,
                expr.span,
                &format!("this `to_owned` call clones the {0} itself and does not cause the {0} contents to become owned", input_type),
                "consider using, depending on intent",
                format!("{0}.clone()` or `{0}.into_owned()", recv_snip),
                app,
            );
            return true;
        }
    }
    false
}
