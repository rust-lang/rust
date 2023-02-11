use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{eager_or_lazy, is_from_proc_macro, usage};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::UNNECESSARY_LAZY_EVALUATIONS;

/// lint use of `<fn>_else(simple closure)` for `Option`s and `Result`s that can be
/// replaced with `<fn>(return value of simple closure)`
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    arg: &'tcx hir::Expr<'_>,
    simplify_using: &str,
) {
    if is_from_proc_macro(cx, expr) {
        return;
    }

    let is_option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Option);
    let is_result = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result);
    let is_bool = cx.typeck_results().expr_ty(recv).is_bool();

    if is_option || is_result || is_bool {
        if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = arg.kind {
            let body = cx.tcx.hir().body(body);
            let body_expr = &body.value;

            if usage::BindingUsageFinder::are_params_used(cx, body) {
                return;
            }

            if eager_or_lazy::switch_to_eager_eval(cx, body_expr) {
                let msg = if is_option {
                    "unnecessary closure used to substitute value for `Option::None`"
                } else if is_result {
                    "unnecessary closure used to substitute value for `Result::Err`"
                } else {
                    "unnecessary closure used with `bool::then`"
                };
                let applicability = if body
                    .params
                    .iter()
                    // bindings are checked to be unused above
                    .all(|param| matches!(param.pat.kind, hir::PatKind::Binding(..) | hir::PatKind::Wild))
                {
                    Applicability::MachineApplicable
                } else {
                    // replacing the lambda may break type inference
                    Applicability::MaybeIncorrect
                };

                // This is a duplicate of what's happening in clippy_lints::methods::method_call,
                // which isn't ideal, We want to get the method call span,
                // but prefer to avoid changing the signature of the function itself.
                if let hir::ExprKind::MethodCall(.., span) = expr.kind {
                    span_lint_and_then(cx, UNNECESSARY_LAZY_EVALUATIONS, expr.span, msg, |diag| {
                        diag.span_suggestion(
                            span,
                            format!("use `{simplify_using}(..)` instead"),
                            format!("{simplify_using}({})", snippet(cx, body_expr.span, "..")),
                            applicability,
                        );
                    });
                }
            }
        }
    }
}
