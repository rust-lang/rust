use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::ty_from_hir_ty;
use clippy_utils::{SpanlessEq, is_in_const_context, is_integer_literal};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for expressions like `x.count_ones() == 1` or `x & (x - 1) == 0`, with x and unsigned integer, which may be manual
    /// reimplementations of `x.is_power_of_two()`.
    ///
    /// ### Why is this bad?
    /// Manual reimplementations of `is_power_of_two` increase code complexity for little benefit.
    ///
    /// ### Example
    /// ```no_run
    /// let a: u32 = 4;
    /// let result = a.count_ones() == 1;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let a: u32 = 4;
    /// let result = a.is_power_of_two();
    /// ```
    #[clippy::version = "1.83.0"]
    pub MANUAL_IS_POWER_OF_TWO,
    pedantic,
    "manually reimplementing `is_power_of_two`"
}

pub struct ManualIsPowerOfTwo {
    msrv: Msrv,
}

impl_lint_pass!(ManualIsPowerOfTwo => [MANUAL_IS_POWER_OF_TWO]);

impl ManualIsPowerOfTwo {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }

    fn build_sugg(&self, cx: &LateContext<'_>, expr: &Expr<'_>, receiver: &Expr<'_>) {
        if is_in_const_context(cx) && !self.msrv.meets(cx, msrvs::CONST_IS_POWER_OF_TWO) {
            return;
        }

        let mut applicability = Applicability::MachineApplicable;
        let snippet = Sugg::hir_with_applicability(cx, receiver, "_", &mut applicability);

        span_lint_and_sugg(
            cx,
            MANUAL_IS_POWER_OF_TWO,
            expr.span,
            "manually reimplementing `is_power_of_two`",
            "consider using `.is_power_of_two()`",
            format!("{}.is_power_of_two()", snippet.maybe_paren()),
            applicability,
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualIsPowerOfTwo {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if !expr.span.from_expansion()
            && let Some((lhs, rhs)) = unexpanded_binop_operands(expr, BinOpKind::Eq)
        {
            if let Some(a) = count_ones_receiver(cx, lhs)
                && is_integer_literal(rhs, 1)
            {
                self.build_sugg(cx, expr, a);
            } else if let Some(a) = count_ones_receiver(cx, rhs)
                && is_integer_literal(lhs, 1)
            {
                self.build_sugg(cx, expr, a);
            } else if is_integer_literal(rhs, 0)
                && let Some(a) = is_and_minus_one(cx, lhs)
            {
                self.build_sugg(cx, expr, a);
            } else if is_integer_literal(lhs, 0)
                && let Some(a) = is_and_minus_one(cx, rhs)
            {
                self.build_sugg(cx, expr, a);
            }
        }
    }
}

/// Return the unsigned integer receiver of `.count_ones()` or the argument of
/// `<int-type>::count_ones(…)`.
fn count_ones_receiver<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    let (method, ty, receiver) = if let ExprKind::MethodCall(method_name, receiver, [], _) = expr.kind {
        (method_name, cx.typeck_results().expr_ty_adjusted(receiver), receiver)
    } else if let ExprKind::Call(func, [arg]) = expr.kind
        && let ExprKind::Path(QPath::TypeRelative(ty, func_name)) = func.kind
    {
        (func_name, ty_from_hir_ty(cx, ty), arg)
    } else {
        return None;
    };
    (method.ident.as_str() == "count_ones" && matches!(ty.kind(), ty::Uint(_))).then_some(receiver)
}

/// Return `greater` if `smaller == greater - 1`
fn is_one_less<'tcx>(
    cx: &LateContext<'tcx>,
    greater: &'tcx Expr<'tcx>,
    smaller: &Expr<'tcx>,
) -> Option<&'tcx Expr<'tcx>> {
    if let Some((lhs, rhs)) = unexpanded_binop_operands(smaller, BinOpKind::Sub)
        && SpanlessEq::new(cx).eq_expr(greater, lhs)
        && is_integer_literal(rhs, 1)
        && matches!(cx.typeck_results().expr_ty_adjusted(greater).kind(), ty::Uint(_))
    {
        Some(greater)
    } else {
        None
    }
}

/// Return `v` if `expr` is `v & (v - 1)` or `(v - 1) & v`
fn is_and_minus_one<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    let (lhs, rhs) = unexpanded_binop_operands(expr, BinOpKind::BitAnd)?;
    is_one_less(cx, lhs, rhs).or_else(|| is_one_less(cx, rhs, lhs))
}

/// Return the operands of the `expr` binary operation if the operator is `op` and none of the
/// operands come from expansion.
fn unexpanded_binop_operands<'hir>(expr: &Expr<'hir>, op: BinOpKind) -> Option<(&'hir Expr<'hir>, &'hir Expr<'hir>)> {
    if let ExprKind::Binary(binop, lhs, rhs) = expr.kind
        && binop.node == op
        && !lhs.span.from_expansion()
        && !rhs.span.from_expansion()
    {
        Some((lhs, rhs))
    } else {
        None
    }
}
