use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_span::{Span, sym};

use super::SUSPICIOUS_TO_OWNED;

pub fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, method_span: Span) -> bool {
    if cx
        .typeck_results()
        .type_dependent_def_id(expr.hir_id)
        .opt_parent(cx)
        .is_diag_item(cx, sym::ToOwned)
        && let input_type = cx.typeck_results().expr_ty(expr)
        && input_type.is_diag_item(cx, sym::Cow)
    {
        let app = Applicability::MaybeIncorrect;
        span_lint_and_then(
            cx,
            SUSPICIOUS_TO_OWNED,
            expr.span,
            with_forced_trimmed_paths!(format!(
                "this `to_owned` call clones the `{input_type}` itself and does not cause its contents to become owned"
            )),
            |diag| {
                diag.span_suggestion(
                    method_span,
                    "depending on intent, either make the `Cow` an `Owned` variant",
                    "into_owned".to_string(),
                    app,
                );
                diag.span_suggestion(method_span, "or clone the `Cow` itself", "clone".to_string(), app);
            },
        );
        return true;
    }
    false
}
