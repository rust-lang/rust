use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::If;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::HasSession as _;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{eq_expr_value, peel_blocks, span_contains_comment};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Ty};
use rustc_session::impl_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects patterns like `if a > b { a - b } else { b - a }` and suggests using `a.abs_diff(b)`.
    ///
    /// ### Why is this bad?
    /// Using `abs_diff` is shorter, more readable, and avoids control flow.
    ///
    /// ### Examples
    /// ```no_run
    /// # let (a, b) = (5_usize, 3_usize);
    /// if a > b {
    ///     a - b
    /// } else {
    ///     b - a
    /// }
    /// # ;
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let (a, b) = (5_usize, 3_usize);
    /// a.abs_diff(b)
    /// # ;
    /// ```
    #[clippy::version = "1.88.0"]
    pub MANUAL_ABS_DIFF,
    complexity,
    "using an if-else pattern instead of `abs_diff`"
}

impl_lint_pass!(ManualAbsDiff => [MANUAL_ABS_DIFF]);

pub struct ManualAbsDiff {
    msrv: Msrv,
}

impl ManualAbsDiff {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for ManualAbsDiff {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !expr.span.from_expansion()
            && let Some(if_expr) = If::hir(expr)
            && let Some(r#else) = if_expr.r#else
            && let ExprKind::Binary(op, rhs, lhs) = if_expr.cond.kind
            && let (BinOpKind::Gt | BinOpKind::Ge, mut a, mut b) | (BinOpKind::Lt | BinOpKind::Le, mut b, mut a) =
                (op.node, rhs, lhs)
            && let Some(ty) = self.are_ty_eligible(cx, a, b)
            && is_sub_expr(cx, if_expr.then, a, b, ty)
            && is_sub_expr(cx, r#else, b, a, ty)
        {
            span_lint_and_then(
                cx,
                MANUAL_ABS_DIFF,
                expr.span,
                "manual absolute difference pattern without using `abs_diff`",
                |diag| {
                    if is_unsuffixed_numeral_lit(a) && !is_unsuffixed_numeral_lit(b) {
                        (a, b) = (b, a);
                    }
                    let applicability = {
                        let source_map = cx.sess().source_map();
                        if span_contains_comment(source_map, if_expr.then.span)
                            || span_contains_comment(source_map, r#else.span)
                        {
                            Applicability::MaybeIncorrect
                        } else {
                            Applicability::MachineApplicable
                        }
                    };
                    let sugg = format!(
                        "{}.abs_diff({})",
                        Sugg::hir(cx, a, "..").maybe_paren(),
                        Sugg::hir(cx, b, "..")
                    );
                    diag.span_suggestion(expr.span, "replace with `abs_diff`", sugg, applicability);
                },
            );
        }
    }
}

impl ManualAbsDiff {
    /// Returns a type if `a` and `b` are both of it, and this lint can be applied to that
    /// type (currently, any primitive int, or a `Duration`)
    fn are_ty_eligible<'tcx>(&self, cx: &LateContext<'tcx>, a: &Expr<'_>, b: &Expr<'_>) -> Option<Ty<'tcx>> {
        let is_int = |ty: Ty<'_>| matches!(ty.kind(), ty::Uint(_) | ty::Int(_)) && self.msrv.meets(cx, msrvs::ABS_DIFF);
        let is_duration =
            |ty| is_type_diagnostic_item(cx, ty, sym::Duration) && self.msrv.meets(cx, msrvs::DURATION_ABS_DIFF);

        let a_ty = cx.typeck_results().expr_ty(a).peel_refs();
        (a_ty == cx.typeck_results().expr_ty(b).peel_refs() && (is_int(a_ty) || is_duration(a_ty))).then_some(a_ty)
    }
}

/// Checks if the given expression is a subtraction operation between two expected expressions,
/// i.e. if `expr` is `{expected_a} - {expected_b}`.
///
/// If `expected_ty` is a signed primitive integer, this function will only return `Some` if the
/// subtraction expr is wrapped in a cast to the equivalent unsigned int.
fn is_sub_expr(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    expected_a: &Expr<'_>,
    expected_b: &Expr<'_>,
    expected_ty: Ty<'_>,
) -> bool {
    let expr = peel_blocks(expr).kind;

    if let ty::Int(ty) = expected_ty.kind() {
        let unsigned = Ty::new_uint(cx.tcx, ty.to_unsigned());

        return if let ExprKind::Cast(expr, cast_ty) = expr
            && cx.typeck_results().node_type(cast_ty.hir_id) == unsigned
        {
            is_sub_expr(cx, expr, expected_a, expected_b, unsigned)
        } else {
            false
        };
    }

    if let ExprKind::Binary(op, a, b) = expr
        && let BinOpKind::Sub = op.node
        && eq_expr_value(cx, a, expected_a)
        && eq_expr_value(cx, b, expected_b)
    {
        true
    } else {
        false
    }
}

fn is_unsuffixed_numeral_lit(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Lit(lit) if lit.node.is_numeric() && lit.node.is_unsuffixed())
}
