use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Lints subtraction between `Instant::now()` and another `Instant`.
    ///
    /// ### Why is this bad?
    /// It is easy to accidentally write `prev_instant - Instant::now()`, which will always be 0ns
    /// as `Instant` subtraction saturates.
    ///
    /// `prev_instant.elapsed()` also more clearly signals intention.
    ///
    /// ### Example
    /// ```rust
    /// use std::time::Instant;
    /// let prev_instant = Instant::now();
    /// let duration = Instant::now() - prev_instant;
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::time::Instant;
    /// let prev_instant = Instant::now();
    /// let duration = prev_instant.elapsed();
    /// ```
    #[clippy::version = "1.64.0"]
    pub MANUAL_INSTANT_ELAPSED,
    pedantic,
    "subtraction between `Instant::now()` and previous `Instant`"
}

declare_lint_pass!(ManualInstantElapsed => [MANUAL_INSTANT_ELAPSED]);

impl LateLintPass<'_> for ManualInstantElapsed {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Binary(Spanned {node: BinOpKind::Sub, ..}, lhs, rhs) = expr.kind
            && check_instant_now_call(cx, lhs)
            && let ty_resolved = cx.typeck_results().expr_ty(rhs)
            && let rustc_middle::ty::Adt(def, _) = ty_resolved.kind()
            && clippy_utils::match_def_path(cx, def.did(), &clippy_utils::paths::INSTANT)
            && let Some(sugg) = clippy_utils::sugg::Sugg::hir_opt(cx, rhs)
        {
            span_lint_and_sugg(
                cx,
                MANUAL_INSTANT_ELAPSED,
                expr.span,
                "manual implementation of `Instant::elapsed`",
                "try",
                format!("{}.elapsed()", sugg.maybe_par()),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn check_instant_now_call(cx: &LateContext<'_>, expr_block: &'_ Expr<'_>) -> bool {
    if let ExprKind::Call(fn_expr, []) = expr_block.kind
        && let Some(fn_id) = clippy_utils::path_def_id(cx, fn_expr)
        && clippy_utils::match_def_path(cx, fn_id, &clippy_utils::paths::INSTANT_NOW)
    {
        true
    } else {
        false
    }
}
