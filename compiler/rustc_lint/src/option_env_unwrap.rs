use rustc_hir::{Expr, ExprKind};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{sym, ExpnKind, MacroKind, Span};

use crate::{
    lints::{OptionEnvUnwrapDiag, OptionEnvUnwrapSuggestion},
    LateContext, LateLintPass, LintContext,
};

declare_lint! {
    /// The `incorrect_option_env_unwraps` lint checks for usage of
    /// `option_env!(...).unwrap()` and suggests using the `env!` macro instead.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// let _ = option_env!("HOME").unwrap();
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Unwrapping the result of `option_env!` will panic
    /// at run-time if the environment variable doesn't exist, whereas `env!`
    /// catches it at compile-time.
    INCORRECT_OPTION_ENV_UNWRAPS,
    Warn,
    "using `option_env!(...).unwrap()` to get environment variable"
}

declare_lint_pass!(OptionEnvUnwrap => [INCORRECT_OPTION_ENV_UNWRAPS]);

impl<'tcx> LateLintPass<'tcx> for OptionEnvUnwrap {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        let ExprKind::MethodCall(_, receiver, _, _) = expr.kind else { return; };

        let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) else { return; };

        if !matches!(
            cx.tcx.get_diagnostic_name(method_def_id),
            Some(sym::option_unwrap | sym::option_expect)
        ) {
            return;
        }

        // Handle found environment variable `Option::Some(...)`
        let caller_span = if let ExprKind::Call(caller, _) = &receiver.kind {
            caller.span
        // Handle not found environment variable `Option::None`
        } else if let ExprKind::Path(qpath) = &receiver.kind {
            qpath.span()
        } else {
            return;
        };

        if is_direct_expn_of_option_env(caller_span) {
            cx.emit_spanned_lint(
                INCORRECT_OPTION_ENV_UNWRAPS,
                expr.span,
                OptionEnvUnwrapDiag {
                    suggestion: extract_inner_macro_call(cx, caller_span)
                        .map(|replace| OptionEnvUnwrapSuggestion { span: expr.span, replace }),
                },
            )
        }
    }
}

fn is_direct_expn_of_option_env(span: Span) -> bool {
    if span.from_expansion() {
        let data = span.ctxt().outer_expn_data();

        if let ExpnKind::Macro(MacroKind::Bang, mac_name) = data.kind {
            return mac_name == sym::option_env;
        }
    }

    false
}

/// Given a Span representing a macro call: `option_env! ( \"j)j\")` get the inner
/// content, ie. ` \"j)j\"`
fn extract_inner_macro_call(cx: &LateContext<'_>, span: Span) -> Option<String> {
    let snippet = cx.sess().parse_sess.source_map().span_to_snippet(span).ok()?;

    let mut inner = snippet.chars().skip_while(|c| !"([{".contains(*c)).collect::<String>();

    // remove last character, ie one of `)]}`
    inner.pop()?;
    // remove first character, ie one of `([{`
    inner.remove(0);

    Some(inner)
}
