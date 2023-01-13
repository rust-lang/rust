use clippy_utils::consts::{constant_full_int, FullInt};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{in_constant, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, Node, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
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
    #[clippy::version = "1.64.0"]
    pub MANUAL_REM_EUCLID,
    complexity,
    "manually reimplementing `rem_euclid`"
}

pub struct ManualRemEuclid {
    msrv: Msrv,
}

impl ManualRemEuclid {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualRemEuclid => [MANUAL_REM_EUCLID]);

impl<'tcx> LateLintPass<'tcx> for ManualRemEuclid {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if !self.msrv.meets(msrvs::REM_EUCLID) {
            return;
        }

        if in_constant(cx, expr.hir_id) && !self.msrv.meets(msrvs::REM_EUCLID_CONST) {
            return;
        }

        if in_external_macro(cx.sess(), expr.span) {
            return;
        }

        if let ExprKind::Binary(op1, expr1, right) = expr.kind
            && op1.node == BinOpKind::Rem
            && let Some(const1) = check_for_unsigned_int_constant(cx, right)
            && let ExprKind::Binary(op2, left, right) = expr1.kind
            && op2.node == BinOpKind::Add
            && let Some((const2, expr2)) = check_for_either_unsigned_int_constant(cx, left, right)
            && let ExprKind::Binary(op3, expr3, right) = expr2.kind
            && op3.node == BinOpKind::Rem
            && let Some(const3) = check_for_unsigned_int_constant(cx, right)
            // Also ensures the const is nonzero since zero can't be a divisor
            && const1 == const2 && const2 == const3
            && let Some(hir_id) = path_to_local(expr3)
            && let Some(Node::Pat(_)) = cx.tcx.hir().find(hir_id) {
                // Apply only to params or locals with annotated types
                match cx.tcx.hir().find_parent(hir_id) {
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

// Checks if either the left or right expressions can be an unsigned int constant and returns that
// constant along with the other expression unchanged if so
fn check_for_either_unsigned_int_constant<'a>(
    cx: &'a LateContext<'_>,
    left: &'a Expr<'_>,
    right: &'a Expr<'_>,
) -> Option<(u128, &'a Expr<'a>)> {
    check_for_unsigned_int_constant(cx, left)
        .map(|int_const| (int_const, right))
        .or_else(|| check_for_unsigned_int_constant(cx, right).map(|int_const| (int_const, left)))
}

fn check_for_unsigned_int_constant<'a>(cx: &'a LateContext<'_>, expr: &'a Expr<'_>) -> Option<u128> {
    let Some(int_const) = constant_full_int(cx, cx.typeck_results(), expr) else { return None };
    match int_const {
        FullInt::S(s) => s.try_into().ok(),
        FullInt::U(u) => Some(u),
    }
}
