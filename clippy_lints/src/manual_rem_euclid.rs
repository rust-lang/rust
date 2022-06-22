use clippy_utils::consts::{constant_full_int, FullInt};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{in_constant, meets_msrv, msrvs, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, Node, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for an expression like `((x % 4) + 4) % 4` which is a common manual reimplementation
    /// of `x.rem_euclid(4)`.
    ///
    /// ### Why is this bad?
    /// It's simpler and more readable.
    ///
    /// ### Example
    /// ```rust
    /// let x: i32 = 24;
    /// let rem = ((x % 4) + 4) % 4;
    /// ```
    /// Use instead:
    /// ```rust
    /// let x: i32 = 24;
    /// let rem = x.rem_euclid(4);
    /// ```
    #[clippy::version = "1.63.0"]
    pub MANUAL_REM_EUCLID,
    complexity,
    "manually reimplementing `rem_euclid`"
}

pub struct ManualRemEuclid {
    msrv: Option<RustcVersion>,
}

impl ManualRemEuclid {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualRemEuclid => [MANUAL_REM_EUCLID]);

impl<'tcx> LateLintPass<'tcx> for ManualRemEuclid {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !meets_msrv(self.msrv, msrvs::REM_EUCLID) {
            return;
        }

        if in_constant(cx, expr.hir_id) && !meets_msrv(self.msrv, msrvs::REM_EUCLID_CONST) {
            return;
        }

        if let ExprKind::Binary(op1, ..) = expr.kind
            && op1.node == BinOpKind::Rem
            && let Some((const1, expr1)) = check_for_positive_int_constant(cx, expr, false)
            && let ExprKind::Binary(op2, ..) = expr1.kind
            && op2.node == BinOpKind::Add
            && let Some((const2, expr2)) = check_for_positive_int_constant(cx, expr1, true)
            && let ExprKind::Binary(op3, ..) = expr2.kind
            && op3.node == BinOpKind::Rem
            && let Some((const3, expr3)) = check_for_positive_int_constant(cx, expr2, false)
            && const1 == const2 && const2 == const3
            && let Some(hir_id) = path_to_local(expr3)
            && let Some(Node::Binding(_)) = cx.tcx.hir().find(hir_id) {
                // Apply only to params or locals with annotated types
                match cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
                    Some(Node::Param(..)) => (),
                    Some(Node::Local(local)) => {
                        let Some(ty) = local.ty else { return };
                        if matches!(ty.kind, TyKind::Infer) {
                            return;
                        }
                    }
                    _ => return,
                };

                let mut app = Applicability::MachineApplicable;
                let rem_of = snippet_with_applicability(cx, expr3.span, "_", &mut app);
                span_lint_and_sugg(
                    cx,
                    MANUAL_REM_EUCLID,
                    expr.span,
                    "manual `rem_euclid` implementation",
                    "consider using",
                    format!("{rem_of}.rem_euclid({const1})"),
                    app,
                );
        }
    }

    extract_msrv_attr!(LateContext);
}

// Takes a binary expression and separates the operands into the int constant and the other
// operand. Ensures the int constant is positive. Operators that are not commutative must have the
// constant appear on the right hand side to return a value.
fn check_for_positive_int_constant<'a>(
    cx: &'a LateContext<'_>,
    expr: &'a Expr<'_>,
    is_commutative: bool,
) -> Option<(u128, &'a Expr<'a>)> {
    let (int_const, other_op) = if let ExprKind::Binary(_, left, right) = expr.kind {
        if let Some(int_const) = constant_full_int(cx, cx.typeck_results(), right) {
            (int_const, left)
        } else if is_commutative && let Some(int_const) = constant_full_int(cx, cx.typeck_results(), left) {
            (int_const, right)
        } else {
            return None;
        }
    } else {
        return None;
    };

    if int_const > FullInt::S(0) {
        let val = match int_const {
            FullInt::S(s) => s.try_into().ok()?,
            FullInt::U(u) => u,
        };
        Some((val, other_op))
    } else {
        None
    }
}
