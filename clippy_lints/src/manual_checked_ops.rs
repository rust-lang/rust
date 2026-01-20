use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::visitors::{Descend, for_each_expr_without_closures};
use clippy_utils::{SpanlessEq, is_integer_literal};
use rustc_hir::{AssignOpKind, BinOpKind, Block, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Detects manual zero checks before dividing integers, such as `if x != 0 { y / x }`.
    ///
    /// ### Why is this bad?
    /// `checked_div` already handles the zero case and makes the intent clearer while avoiding a
    /// panic from a manual division.
    ///
    /// ### Example
    /// ```no_run
    /// # let (a, b) = (10u32, 5u32);
    /// if b != 0 {
    ///     let result = a / b;
    ///     println!("{result}");
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let (a, b) = (10u32, 5u32);
    /// if let Some(result) = a.checked_div(b) {
    ///     println!("{result}");
    /// }
    /// ```
    #[clippy::version = "1.95.0"]
    pub MANUAL_CHECKED_OPS,
    complexity,
    "manual zero checks before dividing integers"
}
declare_lint_pass!(ManualCheckedOps => [MANUAL_CHECKED_OPS]);

#[derive(Copy, Clone)]
enum NonZeroBranch {
    Then,
    Else,
}

impl LateLintPass<'_> for ManualCheckedOps {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::If(cond, then, r#else) = expr.kind
            && !expr.span.from_expansion()
            && let Some((divisor, branch)) = divisor_from_condition(cond)
            // This lint is intended for unsigned integers only.
            //
            // For signed integers, the most direct refactor to `checked_div` is often not
            // semantically equivalent to the original guard. For example, `rhs > 0` deliberately
            // excludes negative divisors, while `checked_div` would return `Some` for `rhs = -2`.
            // Also, `checked_div` can return `None` for `MIN / -1`, which requires additional
            // handling beyond the zero check.
            && is_unsigned_integer(cx, divisor)
            && let Some(block) = branch_block(then, r#else, branch)
        {
            let mut eq = SpanlessEq::new(cx).deny_side_effects().paths_by_resolution();
            if !eq.eq_expr(divisor, divisor) {
                return;
            }

            let mut division_spans = Vec::new();
            let mut first_use = None;

            let found_early_use = for_each_expr_without_closures(block, |e| {
                if let ExprKind::Binary(binop, lhs, rhs) = e.kind
                    && binop.node == BinOpKind::Div
                    && eq.eq_expr(rhs, divisor)
                    && is_unsigned_integer(cx, lhs)
                {
                    match first_use {
                        None => first_use = Some(UseKind::Division),
                        Some(UseKind::Other) => return ControlFlow::Break(()),
                        Some(UseKind::Division) => {},
                    }

                    division_spans.push(e.span);

                    ControlFlow::<(), _>::Continue(Descend::No)
                } else if let ExprKind::AssignOp(op, lhs, rhs) = e.kind
                    && op.node == AssignOpKind::DivAssign
                    && eq.eq_expr(rhs, divisor)
                    && is_unsigned_integer(cx, lhs)
                {
                    match first_use {
                        None => first_use = Some(UseKind::Division),
                        Some(UseKind::Other) => return ControlFlow::Break(()),
                        Some(UseKind::Division) => {},
                    }

                    division_spans.push(e.span);

                    ControlFlow::<(), _>::Continue(Descend::No)
                } else if eq.eq_expr(e, divisor) {
                    if first_use.is_none() {
                        first_use = Some(UseKind::Other);
                        return ControlFlow::Break(());
                    }
                    ControlFlow::<(), _>::Continue(Descend::Yes)
                } else {
                    ControlFlow::<(), _>::Continue(Descend::Yes)
                }
            });

            if found_early_use.is_some() || first_use != Some(UseKind::Division) || division_spans.is_empty() {
                return;
            }

            span_lint_and_then(cx, MANUAL_CHECKED_OPS, cond.span, "manual checked division", |diag| {
                diag.span_label(cond.span, "check performed here");
                if let Some((first, rest)) = division_spans.split_first() {
                    diag.span_label(*first, "division performed here");
                    if !rest.is_empty() {
                        diag.span_labels(rest.to_vec(), "... and here");
                    }
                }
                diag.help("consider using `checked_div`");
            });
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum UseKind {
    Division,
    Other,
}

fn divisor_from_condition<'tcx>(cond: &'tcx Expr<'tcx>) -> Option<(&'tcx Expr<'tcx>, NonZeroBranch)> {
    let ExprKind::Binary(binop, lhs, rhs) = cond.kind else {
        return None;
    };

    match binop.node {
        BinOpKind::Ne | BinOpKind::Lt if is_zero(lhs) => Some((rhs, NonZeroBranch::Then)),
        BinOpKind::Ne | BinOpKind::Gt if is_zero(rhs) => Some((lhs, NonZeroBranch::Then)),
        BinOpKind::Eq if is_zero(lhs) => Some((rhs, NonZeroBranch::Else)),
        BinOpKind::Eq if is_zero(rhs) => Some((lhs, NonZeroBranch::Else)),
        _ => None,
    }
}

fn branch_block<'tcx>(
    then: &'tcx Expr<'tcx>,
    r#else: Option<&'tcx Expr<'tcx>>,
    branch: NonZeroBranch,
) -> Option<&'tcx Block<'tcx>> {
    match branch {
        NonZeroBranch::Then => match then.kind {
            ExprKind::Block(block, _) => Some(block),
            _ => None,
        },
        NonZeroBranch::Else => match r#else?.kind {
            ExprKind::Block(block, _) => Some(block),
            _ => None,
        },
    }
}

fn is_zero(expr: &Expr<'_>) -> bool {
    is_integer_literal(expr, 0)
}

fn is_unsigned_integer(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    matches!(cx.typeck_results().expr_ty(expr).peel_refs().kind(), ty::Uint(_))
}
