use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::source_map::Spanned;
use rustc_span::sym;

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
    /// ```no_run
    /// use std::time::Instant;
    /// let prev_instant = Instant::now();
    /// let duration = Instant::now() - prev_instant;
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::time::Instant;
    /// let prev_instant = Instant::now();
    /// let duration = prev_instant.elapsed();
    /// ```
    #[clippy::version = "1.65.0"]
    pub MANUAL_INSTANT_ELAPSED,
    pedantic,
    "subtraction between `Instant::now()` and previous `Instant`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Lints subtraction between an `Instant` and a `Duration`.
    ///
    /// ### Why is this bad?
    /// Unchecked subtraction could cause underflow on certain platforms, leading to
    /// unintentional panics.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now() - Duration::from_secs(5);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now().checked_sub(Duration::from_secs(5));
    /// ```
    #[clippy::version = "1.67.0"]
    pub UNCHECKED_DURATION_SUBTRACTION,
    pedantic,
    "finds unchecked subtraction of a 'Duration' from an 'Instant'"
}

pub struct InstantSubtraction {
    msrv: Msrv,
}

impl InstantSubtraction {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(InstantSubtraction => [MANUAL_INSTANT_ELAPSED, UNCHECKED_DURATION_SUBTRACTION]);

impl LateLintPass<'_> for InstantSubtraction {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Sub, ..
            },
            lhs,
            rhs,
        ) = expr.kind
            && let typeck = cx.typeck_results()
            && ty::is_type_diagnostic_item(cx, typeck.expr_ty(lhs), sym::Instant)
        {
            let rhs_ty = typeck.expr_ty(rhs);

            if is_instant_now_call(cx, lhs)
                && ty::is_type_diagnostic_item(cx, rhs_ty, sym::Instant)
                && let Some(sugg) = Sugg::hir_opt(cx, rhs)
            {
                print_manual_instant_elapsed_sugg(cx, expr, sugg);
            } else if ty::is_type_diagnostic_item(cx, rhs_ty, sym::Duration)
                && !expr.span.from_expansion()
                && self.msrv.meets(cx, msrvs::TRY_FROM)
            {
                print_unchecked_duration_subtraction_sugg(cx, lhs, rhs, expr);
            }
        }
    }
}

fn is_instant_now_call(cx: &LateContext<'_>, expr_block: &'_ Expr<'_>) -> bool {
    if let ExprKind::Call(fn_expr, []) = expr_block.kind
        && let Some(fn_id) = clippy_utils::path_def_id(cx, fn_expr)
        && cx.tcx.is_diagnostic_item(sym::instant_now, fn_id)
    {
        true
    } else {
        false
    }
}

fn print_manual_instant_elapsed_sugg(cx: &LateContext<'_>, expr: &Expr<'_>, sugg: Sugg<'_>) {
    span_lint_and_sugg(
        cx,
        MANUAL_INSTANT_ELAPSED,
        expr.span,
        "manual implementation of `Instant::elapsed`",
        "try",
        format!("{}.elapsed()", sugg.maybe_paren()),
        Applicability::MachineApplicable,
    );
}

fn print_unchecked_duration_subtraction_sugg(
    cx: &LateContext<'_>,
    left_expr: &Expr<'_>,
    right_expr: &Expr<'_>,
    expr: &Expr<'_>,
) {
    let mut applicability = Applicability::MachineApplicable;

    let ctxt = expr.span.ctxt();
    let left_expr = snippet_with_context(cx, left_expr.span, ctxt, "<instant>", &mut applicability).0;
    let right_expr = snippet_with_context(cx, right_expr.span, ctxt, "<duration>", &mut applicability).0;

    span_lint_and_sugg(
        cx,
        UNCHECKED_DURATION_SUBTRACTION,
        expr.span,
        "unchecked subtraction of a 'Duration' from an 'Instant'",
        "try",
        format!("{left_expr}.checked_sub({right_expr}).unwrap()"),
        applicability,
    );
}
