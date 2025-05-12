use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg::{Sugg, has_enclosing_paren};
use clippy_utils::{SpanlessEq, sym};
use rustc_ast::{BinOpKind, LitIntType, LitKind, UnOp};
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self};
use rustc_session::impl_lint_pass;
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for an expression like `(x + (y - 1)) / y` which is a common manual reimplementation
    /// of `x.div_ceil(y)`.
    ///
    /// ### Why is this bad?
    /// It's simpler, clearer and more readable.
    ///
    /// ### Example
    /// ```no_run
    /// let x: i32 = 7;
    /// let y: i32 = 4;
    /// let div = (x + (y - 1)) / y;
    /// ```
    /// Use instead:
    /// ```no_run
    /// #![feature(int_roundings)]
    /// let x: i32 = 7;
    /// let y: i32 = 4;
    /// let div = x.div_ceil(y);
    /// ```
    #[clippy::version = "1.83.0"]
    pub MANUAL_DIV_CEIL,
    complexity,
    "manually reimplementing `div_ceil`"
}

pub struct ManualDivCeil {
    msrv: Msrv,
}

impl ManualDivCeil {
    #[must_use]
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(ManualDivCeil => [MANUAL_DIV_CEIL]);

impl<'tcx> LateLintPass<'tcx> for ManualDivCeil {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        let mut applicability = Applicability::MachineApplicable;

        if let ExprKind::Binary(div_op, div_lhs, div_rhs) = expr.kind
            && div_op.node == BinOpKind::Div
            && check_int_ty_and_feature(cx, div_lhs)
            && check_int_ty_and_feature(cx, div_rhs)
            && let ExprKind::Binary(inner_op, inner_lhs, inner_rhs) = div_lhs.kind
            && self.msrv.meets(cx, msrvs::DIV_CEIL)
        {
            // (x + (y - 1)) / y
            if let ExprKind::Binary(sub_op, sub_lhs, sub_rhs) = inner_rhs.kind
                && inner_op.node == BinOpKind::Add
                && sub_op.node == BinOpKind::Sub
                && check_literal(sub_rhs)
                && check_eq_expr(cx, sub_lhs, div_rhs)
            {
                build_suggestion(cx, expr, inner_lhs, div_rhs, &mut applicability);
                return;
            }

            // ((y - 1) + x) / y
            if let ExprKind::Binary(sub_op, sub_lhs, sub_rhs) = inner_lhs.kind
                && inner_op.node == BinOpKind::Add
                && sub_op.node == BinOpKind::Sub
                && check_literal(sub_rhs)
                && check_eq_expr(cx, sub_lhs, div_rhs)
            {
                build_suggestion(cx, expr, inner_rhs, div_rhs, &mut applicability);
                return;
            }

            // (x + y - 1) / y
            if let ExprKind::Binary(add_op, add_lhs, add_rhs) = inner_lhs.kind
                && inner_op.node == BinOpKind::Sub
                && add_op.node == BinOpKind::Add
                && check_literal(inner_rhs)
                && check_eq_expr(cx, add_rhs, div_rhs)
            {
                build_suggestion(cx, expr, add_lhs, div_rhs, &mut applicability);
            }

            // (x + (Y - 1)) / Y
            if inner_op.node == BinOpKind::Add && differ_by_one(inner_rhs, div_rhs) {
                build_suggestion(cx, expr, inner_lhs, div_rhs, &mut applicability);
            }

            // ((Y - 1) + x) / Y
            if inner_op.node == BinOpKind::Add && differ_by_one(inner_lhs, div_rhs) {
                build_suggestion(cx, expr, inner_rhs, div_rhs, &mut applicability);
            }

            // (x - (-Y - 1)) / Y
            if inner_op.node == BinOpKind::Sub
                && let ExprKind::Unary(UnOp::Neg, abs_div_rhs) = div_rhs.kind
                && differ_by_one(abs_div_rhs, inner_rhs)
            {
                build_suggestion(cx, expr, inner_lhs, div_rhs, &mut applicability);
            }
        }
    }
}

/// Checks if two expressions represent non-zero integer literals such that `small_expr + 1 ==
/// large_expr`.
fn differ_by_one(small_expr: &Expr<'_>, large_expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(small) = small_expr.kind
        && let ExprKind::Lit(large) = large_expr.kind
        && let LitKind::Int(s, _) = small.node
        && let LitKind::Int(l, _) = large.node
    {
        Some(l.get()) == s.get().checked_add(1)
    } else if let ExprKind::Unary(UnOp::Neg, small_inner_expr) = small_expr.kind
        && let ExprKind::Unary(UnOp::Neg, large_inner_expr) = large_expr.kind
    {
        differ_by_one(large_inner_expr, small_inner_expr)
    } else {
        false
    }
}

fn check_int_ty_and_feature(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let expr_ty = cx.typeck_results().expr_ty(expr);
    match expr_ty.peel_refs().kind() {
        ty::Uint(_) => true,
        ty::Int(_) => cx.tcx.features().enabled(sym::int_roundings),
        _ => false,
    }
}

fn check_literal(expr: &Expr<'_>) -> bool {
    if let ExprKind::Lit(lit) = expr.kind
        && let LitKind::Int(Pu128(1), _) = lit.node
    {
        return true;
    }
    false
}

fn check_eq_expr(cx: &LateContext<'_>, lhs: &Expr<'_>, rhs: &Expr<'_>) -> bool {
    SpanlessEq::new(cx).eq_expr(lhs, rhs)
}

fn build_suggestion(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    lhs: &Expr<'_>,
    rhs: &Expr<'_>,
    applicability: &mut Applicability,
) {
    let dividend_sugg = Sugg::hir_with_applicability(cx, lhs, "..", applicability).maybe_paren();
    let type_suffix = if cx.typeck_results().expr_ty(lhs).is_numeric()
        && matches!(
            lhs.kind,
            ExprKind::Lit(Spanned {
                node: LitKind::Int(_, LitIntType::Unsuffixed),
                ..
            }) | ExprKind::Unary(
                UnOp::Neg,
                Expr {
                    kind: ExprKind::Lit(Spanned {
                        node: LitKind::Int(_, LitIntType::Unsuffixed),
                        ..
                    }),
                    ..
                }
            )
        ) {
        format!("_{}", cx.typeck_results().expr_ty(rhs))
    } else {
        String::new()
    };
    let dividend_sugg_str = dividend_sugg.into_string();
    // If `dividend_sugg` has enclosing paren like `(-2048)` and we need to add type suffix in the
    // suggestion message, we want to make a suggestion string before `div_ceil` like
    // `(-2048_{type_suffix})`.
    let suggestion_before_div_ceil = if has_enclosing_paren(&dividend_sugg_str) {
        format!(
            "{}{})",
            &dividend_sugg_str[..dividend_sugg_str.len() - 1].to_string(),
            type_suffix
        )
    } else {
        format!("{dividend_sugg_str}{type_suffix}")
    };
    let divisor_snippet = snippet_with_context(cx, rhs.span, expr.span.ctxt(), "..", applicability);

    let sugg = format!("{suggestion_before_div_ceil}.div_ceil({})", divisor_snippet.0);

    span_lint_and_sugg(
        cx,
        MANUAL_DIV_CEIL,
        expr.span,
        "manually reimplementing `div_ceil`",
        "consider using `.div_ceil()`",
        sugg,
        *applicability,
    );
}
