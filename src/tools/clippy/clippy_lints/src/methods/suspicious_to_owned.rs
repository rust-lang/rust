use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::is_diag_trait_item;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::{self};
use rustc_span::sym;

use super::SUSPICIOUS_TO_OWNED;

pub fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) -> bool {
    if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && is_diag_trait_item(cx, method_def_id, sym::ToOwned)
        && let input_type = cx.typeck_results().expr_ty(expr)
        && let ty::Adt(adt, _) = cx.typeck_results().expr_ty(expr).kind()
        && cx.tcx.is_diagnostic_item(sym::Cow, adt.did())
    {
        let mut app = Applicability::MaybeIncorrect;
        let recv_snip = snippet_with_context(cx, recv.span, expr.span.ctxt(), "..", &mut app).0;
        span_lint_and_then(
            cx,
            SUSPICIOUS_TO_OWNED,
            expr.span,
            with_forced_trimmed_paths!(format!(
                "this `to_owned` call clones the {input_type} itself and does not cause the {input_type} contents to become owned"
            )),
            |diag| {
                diag.span_suggestion(
                    expr.span,
                    "depending on intent, either make the Cow an Owned variant",
                    format!("{recv_snip}.into_owned()"),
                    app,
                );
                diag.span_suggestion(
                    expr.span,
                    "or clone the Cow itself",
                    format!("{recv_snip}.clone()"),
                    app,
                );
            },
        );
        return true;
    }
    false
}
