use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::{eager_or_lazy, is_from_proc_macro, usage};
use hir::FnRetTy;
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
    use_turbofish: bool,
) -> bool {
    let is_option = cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Option);
    let is_result = cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Result);
    let is_bool = cx.typeck_results().expr_ty(recv).is_bool();

    if (is_option || is_result || is_bool)
        && let hir::ExprKind::Closure(&hir::Closure {
            body,
            fn_decl,
            kind: hir::ClosureKind::Closure,
            ..
        }) = arg.kind
    {
        let body = cx.tcx.hir_body(body);
        let body_expr = &body.value;

        if usage::BindingUsageFinder::are_params_used(cx, body) || is_from_proc_macro(cx, expr) {
            return false;
        }

        if eager_or_lazy::switch_to_eager_eval(cx, body_expr) {
            let msg = if is_option {
                "unnecessary closure used to substitute value for `Option::None`"
            } else if is_result {
                "unnecessary closure used to substitute value for `Result::Err`"
            } else {
                "unnecessary closure used with `bool::then`"
            };
            let mut applicability = Applicability::MachineApplicable;
            let (ascription, turbofish) = if body
                .params
                .iter()
                // bindings are checked to be unused above
                .all(|param| matches!(param.pat.kind, hir::PatKind::Binding(..) | hir::PatKind::Wild))
            {
                match fn_decl.output {
                    FnRetTy::DefaultReturn(_)
                    | FnRetTy::Return(hir::Ty {
                        kind: hir::TyKind::Infer(()),
                        ..
                    }) => {
                        // type ascription is definitely not needed here
                        (String::new(), String::new())
                    },
                    FnRetTy::Return(ty) => {
                        // explicit type was given on the closure
                        // try to use turbofish for this, since it's less dangerous,
                        // but, failing that, use `as`
                        let ty = snippet_with_applicability(cx, ty.span, "_", &mut applicability);
                        if use_turbofish {
                            (String::new(), format!("::<{ty}>"))
                        } else {
                            (format!(" as {ty}"), String::new())
                        }
                    },
                }
            } else {
                // can't infer the actual type
                applicability = Applicability::MaybeIncorrect;
                (String::new(), String::new())
            };

            // This is a duplicate of what's happening in clippy_lints::methods::method_call,
            // which isn't ideal, We want to get the method call span,
            // but prefer to avoid changing the signature of the function itself.
            if let hir::ExprKind::MethodCall(.., span) = expr.kind {
                span_lint_and_then(cx, UNNECESSARY_LAZY_EVALUATIONS, expr.span, msg, |diag| {
                    diag.span_suggestion_verbose(
                        span,
                        format!("use `{simplify_using}` instead"),
                        format!(
                            "{simplify_using}{turbofish}({}{ascription})",
                            snippet(cx, body_expr.span, "..")
                        ),
                        applicability,
                    );
                });
                return true;
            }
        }
    }
    false
}
