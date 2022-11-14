use clippy_utils::{
    diagnostics::{self, span_lint_and_sugg},
    meets_msrv, msrvs, source,
    sugg::Sugg,
    ty,
};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{source_map::Spanned, sym};

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
    #[clippy::version = "1.65.0"]
    pub MANUAL_INSTANT_ELAPSED,
    pedantic,
    "subtraction between `Instant::now()` and previous `Instant`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Lints subtraction between an [`Instant`] and a [`Duration`].
    ///
    /// ### Why is this bad?
    /// Unchecked subtraction could cause underflow on certain platforms, leading to
    /// unintentional panics.
    ///
    /// ### Example
    /// ```rust
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now() - Duration::from_secs(5);
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # use std::time::{Instant, Duration};
    /// let time_passed = Instant::now().checked_sub(Duration::from_secs(5));
    /// ```
    ///
    /// [`Duration`]: std::time::Duration
    /// [`Instant::now()`]: std::time::Instant::now;
    #[clippy::version = "1.65.0"]
    pub UNCHECKED_DURATION_SUBTRACTION,
    suspicious,
    "finds unchecked subtraction of a 'Duration' from an 'Instant'"
}

pub struct InstantSubtraction {
    msrv: Option<RustcVersion>,
}

impl InstantSubtraction {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
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
        {
            if_chain! {
                if is_instant_now_call(cx, lhs);

                if is_an_instant(cx, rhs);
                if let Some(sugg) = Sugg::hir_opt(cx, rhs);

                then {
                    print_manual_instant_elapsed_sugg(cx, expr, sugg)
                } else {
                    if_chain! {
                        if !expr.span.from_expansion();
                        if meets_msrv(self.msrv, msrvs::TRY_FROM);

                        if is_an_instant(cx, lhs);
                        if is_a_duration(cx, rhs);

                        then {
                            print_unchecked_duration_subtraction_sugg(cx, lhs, rhs, expr)
                        }
                    }
                }
            }
        }
    }

    extract_msrv_attr!(LateContext);
}

fn is_instant_now_call(cx: &LateContext<'_>, expr_block: &'_ Expr<'_>) -> bool {
    if let ExprKind::Call(fn_expr, []) = expr_block.kind
        && let Some(fn_id) = clippy_utils::path_def_id(cx, fn_expr)
        && clippy_utils::match_def_path(cx, fn_id, &clippy_utils::paths::INSTANT_NOW)
    {
        true
    } else {
        false
    }
}

fn is_an_instant(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr);

    match expr_ty.kind() {
        rustc_middle::ty::Adt(def, _) => clippy_utils::match_def_path(cx, def.did(), &clippy_utils::paths::INSTANT),
        _ => false,
    }
}

fn is_a_duration(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr);
    ty::is_type_diagnostic_item(cx, expr_ty, sym::Duration)
}

fn print_manual_instant_elapsed_sugg(cx: &LateContext<'_>, expr: &Expr<'_>, sugg: Sugg<'_>) {
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

fn print_unchecked_duration_subtraction_sugg(
    cx: &LateContext<'_>,
    left_expr: &Expr<'_>,
    right_expr: &Expr<'_>,
    expr: &Expr<'_>,
) {
    let mut applicability = Applicability::MachineApplicable;

    let left_expr =
        source::snippet_with_applicability(cx, left_expr.span, "std::time::Instant::now()", &mut applicability);
    let right_expr = source::snippet_with_applicability(
        cx,
        right_expr.span,
        "std::time::Duration::from_secs(1)",
        &mut applicability,
    );

    diagnostics::span_lint_and_sugg(
        cx,
        UNCHECKED_DURATION_SUBTRACTION,
        expr.span,
        "unchecked subtraction of a 'Duration' from an 'Instant'",
        "try",
        format!("{left_expr}.checked_sub({right_expr}).unwrap()"),
        applicability,
    );
}
