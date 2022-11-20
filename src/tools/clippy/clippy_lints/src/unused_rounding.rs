use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::{Expr, ExprKind, MethodCall};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Detects cases where a whole-number literal float is being rounded, using
    /// the `floor`, `ceil`, or `round` methods.
    ///
    /// ### Why is this bad?
    ///
    /// This is unnecessary and confusing to the reader. Doing this is probably a mistake.
    ///
    /// ### Example
    /// ```rust
    /// let x = 1f32.ceil();
    /// ```
    /// Use instead:
    /// ```rust
    /// let x = 1f32;
    /// ```
    #[clippy::version = "1.63.0"]
    pub UNUSED_ROUNDING,
    nursery,
    "Uselessly rounding a whole number floating-point literal"
}
declare_lint_pass!(UnusedRounding => [UNUSED_ROUNDING]);

fn is_useless_rounding(expr: &Expr) -> Option<(&str, String)> {
    if let ExprKind::MethodCall(box MethodCall { seg, receiver, .. }) = &expr.kind
        && let method_name = seg.ident.name.as_str()
        && (method_name == "ceil" || method_name == "round" || method_name == "floor")
        && let ExprKind::Lit(token_lit) = &receiver.kind
        && token_lit.is_semantic_float() {
            let f = token_lit.symbol.as_str().parse::<f64>().unwrap();
            let mut f_str = token_lit.symbol.to_string();
            match token_lit.suffix {
                Some(suffix) => f_str.push_str(suffix.as_str()),
                None => {}
            }
            if f.fract() == 0.0 {
                Some((method_name, f_str))
            } else {
                None
            }
        } else {
            None
        }
}

impl EarlyLintPass for UnusedRounding {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let Some((method_name, float)) = is_useless_rounding(expr) {
            span_lint_and_sugg(
                cx,
                UNUSED_ROUNDING,
                expr.span,
                &format!("used the `{method_name}` method with a whole number float"),
                &format!("remove the `{method_name}` method call"),
                float,
                Applicability::MachineApplicable,
            );
        }
    }
}
