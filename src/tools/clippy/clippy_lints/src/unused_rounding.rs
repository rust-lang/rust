use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::sym;
use rustc_ast::ast::{Expr, ExprKind, MethodCall};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Symbol;

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
    /// ```no_run
    /// let x = 1f32.ceil();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x = 1f32;
    /// ```
    #[clippy::version = "1.63.0"]
    pub UNUSED_ROUNDING,
    nursery,
    "Uselessly rounding a whole number floating-point literal"
}
declare_lint_pass!(UnusedRounding => [UNUSED_ROUNDING]);

fn is_useless_rounding(cx: &EarlyContext<'_>, expr: &Expr) -> Option<(Symbol, String)> {
    if let ExprKind::MethodCall(box MethodCall {
        seg: name_ident,
        receiver,
        ..
    }) = &expr.kind
        && let method_name = name_ident.ident.name
        && matches!(method_name, sym::ceil | sym::floor | sym::round)
        && let ExprKind::Lit(token_lit) = &receiver.kind
        && token_lit.is_semantic_float()
        && let Ok(f) = token_lit.symbol.as_str().replace('_', "").parse::<f64>()
        && f.fract() == 0.0
    {
        Some((method_name, snippet(cx, receiver.span, "..").into()))
    } else {
        None
    }
}

impl EarlyLintPass for UnusedRounding {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        if let Some((method_name, float)) = is_useless_rounding(cx, expr) {
            span_lint_and_sugg(
                cx,
                UNUSED_ROUNDING,
                expr.span,
                format!("used the `{method_name}` method with a whole number float"),
                format!("remove the `{method_name}` method call"),
                float,
                Applicability::MachineApplicable,
            );
        }
    }
}
