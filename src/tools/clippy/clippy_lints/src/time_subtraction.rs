use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::sugg::Sugg;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
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
    /// Lints subtraction between an `Instant` and a `Duration`, or between two `Duration` values.
    ///
    /// ### Why is this bad?
    /// Unchecked subtraction could cause underflow on certain platforms, leading to
    /// unintentional panics.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now() - Duration::from_secs(5);
    /// let dur1 = Duration::from_secs(3);
    /// let dur2 = Duration::from_secs(5);
    /// let diff = dur1 - dur2;
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now().checked_sub(Duration::from_secs(5));
    /// let dur1 = Duration::from_secs(3);
    /// let dur2 = Duration::from_secs(5);
    /// let diff = dur1.checked_sub(dur2);
    /// ```
    #[clippy::version = "1.67.0"]
    pub UNCHECKED_TIME_SUBTRACTION,
    pedantic,
    "finds unchecked subtraction involving 'Duration' or 'Instant'"
}

pub struct UncheckedTimeSubtraction {
    msrv: Msrv,
}

impl UncheckedTimeSubtraction {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(UncheckedTimeSubtraction => [MANUAL_INSTANT_ELAPSED, UNCHECKED_TIME_SUBTRACTION]);

impl LateLintPass<'_> for UncheckedTimeSubtraction {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Sub, ..
            },
            lhs,
            rhs,
        ) = expr.kind
        {
            let typeck = cx.typeck_results();
            let lhs_ty = typeck.expr_ty(lhs);
            let rhs_ty = typeck.expr_ty(rhs);

            if lhs_ty.is_diag_item(cx, sym::Instant) {
                // Instant::now() - instant
                if is_instant_now_call(cx, lhs)
                    && rhs_ty.is_diag_item(cx, sym::Instant)
                    && let Some(sugg) = Sugg::hir_opt(cx, rhs)
                {
                    print_manual_instant_elapsed_sugg(cx, expr, sugg);
                }
                // instant - duration
                else if rhs_ty.is_diag_item(cx, sym::Duration)
                    && !expr.span.from_expansion()
                    && self.msrv.meets(cx, msrvs::TRY_FROM)
                {
                    // For chained subtraction like (instant - dur1) - dur2, avoid suggestions
                    if is_chained_time_subtraction(cx, lhs) {
                        span_lint(
                            cx,
                            UNCHECKED_TIME_SUBTRACTION,
                            expr.span,
                            "unchecked subtraction of a 'Duration' from an 'Instant'",
                        );
                    } else {
                        // instant - duration
                        print_unchecked_duration_subtraction_sugg(cx, lhs, rhs, expr);
                    }
                }
            } else if lhs_ty.is_diag_item(cx, sym::Duration)
                && rhs_ty.is_diag_item(cx, sym::Duration)
                && !expr.span.from_expansion()
                && self.msrv.meets(cx, msrvs::TRY_FROM)
            {
                // For chained subtraction like (dur1 - dur2) - dur3, avoid suggestions
                if is_chained_time_subtraction(cx, lhs) {
                    span_lint(
                        cx,
                        UNCHECKED_TIME_SUBTRACTION,
                        expr.span,
                        "unchecked subtraction between 'Duration' values",
                    );
                } else {
                    // duration - duration
                    print_unchecked_duration_subtraction_sugg(cx, lhs, rhs, expr);
                }
            }
        }
    }
}

fn is_instant_now_call(cx: &LateContext<'_>, expr_block: &'_ Expr<'_>) -> bool {
    if let ExprKind::Call(fn_expr, []) = expr_block.kind
        && cx.ty_based_def(fn_expr).is_diag_item(cx, sym::instant_now)
    {
        true
    } else {
        false
    }
}

/// Returns true if this subtraction is part of a chain like `(a - b) - c`
fn is_chained_time_subtraction(cx: &LateContext<'_>, lhs: &Expr<'_>) -> bool {
    if let ExprKind::Binary(op, inner_lhs, inner_rhs) = &lhs.kind
        && matches!(op.node, BinOpKind::Sub)
    {
        let typeck = cx.typeck_results();
        let left_ty = typeck.expr_ty(inner_lhs);
        let right_ty = typeck.expr_ty(inner_rhs);
        is_time_type(cx, left_ty) && is_time_type(cx, right_ty)
    } else {
        false
    }
}

/// Returns true if the type is Duration or Instant
fn is_time_type(cx: &LateContext<'_>, ty: Ty<'_>) -> bool {
    ty.is_diag_item(cx, sym::Duration) || ty.is_diag_item(cx, sym::Instant)
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
    let typeck = cx.typeck_results();
    let left_ty = typeck.expr_ty(left_expr);

    let lint_msg = if left_ty.is_diag_item(cx, sym::Instant) {
        "unchecked subtraction of a 'Duration' from an 'Instant'"
    } else {
        "unchecked subtraction between 'Duration' values"
    };

    let mut applicability = Applicability::MachineApplicable;
    let left_sugg = Sugg::hir_with_applicability(cx, left_expr, "<left>", &mut applicability);
    let right_sugg = Sugg::hir_with_applicability(cx, right_expr, "<right>", &mut applicability);

    span_lint_and_sugg(
        cx,
        UNCHECKED_TIME_SUBTRACTION,
        expr.span,
        lint_msg,
        "try",
        format!("{}.checked_sub({}).unwrap()", left_sugg.maybe_paren(), right_sugg),
        applicability,
    );
}
