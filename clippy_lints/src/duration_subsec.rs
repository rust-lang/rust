use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::match_type;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::paths;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for calculation of subsecond microseconds or milliseconds
    /// from other `Duration` methods.
    ///
    /// ### Why is this bad?
    /// It's more concise to call `Duration::subsec_micros()` or
    /// `Duration::subsec_millis()` than to calculate them.
    ///
    /// ### Example
    /// ```rust
    /// # use std::time::Duration;
    /// let dur = Duration::new(5, 0);
    ///
    /// // Bad
    /// let _micros = dur.subsec_nanos() / 1_000;
    /// let _millis = dur.subsec_nanos() / 1_000_000;
    ///
    /// // Good
    /// let _micros = dur.subsec_micros();
    /// let _millis = dur.subsec_millis();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DURATION_SUBSEC,
    complexity,
    "checks for calculation of subsecond microseconds or milliseconds"
}

declare_lint_pass!(DurationSubsec => [DURATION_SUBSEC]);

impl<'tcx> LateLintPass<'tcx> for DurationSubsec {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Binary(Spanned { node: BinOpKind::Div, .. }, left, right) = expr.kind;
            if let ExprKind::MethodCall(method_path, args, _) = left.kind;
            if match_type(cx, cx.typeck_results().expr_ty(&args[0]).peel_refs(), &paths::DURATION);
            if let Some((Constant::Int(divisor), _)) = constant(cx, cx.typeck_results(), right);
            then {
                let suggested_fn = match (method_path.ident.as_str(), divisor) {
                    ("subsec_micros", 1_000) | ("subsec_nanos", 1_000_000) => "subsec_millis",
                    ("subsec_nanos", 1_000) => "subsec_micros",
                    _ => return,
                };
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    DURATION_SUBSEC,
                    expr.span,
                    &format!("calling `{}()` is more concise than this calculation", suggested_fn),
                    "try",
                    format!(
                        "{}.{}()",
                        snippet_with_applicability(cx, args[0].span, "_", &mut applicability),
                        suggested_fn
                    ),
                    applicability,
                );
            }
        }
    }
}
