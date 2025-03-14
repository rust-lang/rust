use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, FullInt};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::{is_in_const_context, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, Node, TyKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for an expression like `((x % 4) + 4) % 4` which is a common manual reimplementation
    /// of `x.rem_euclid(4)`.
    ///
    /// ### Why is this bad?
    /// It's simpler and more readable.
    ///
    /// ### Example
    /// ```no_run
    /// let x: i32 = 24;
    /// let rem = ((x % 4) + 4) % 4;
    /// ```
    /// Use instead:
    /// ```no_run
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
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(ManualRemEuclid => [MANUAL_REM_EUCLID]);

impl<'tcx> LateLintPass<'tcx> for ManualRemEuclid {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        // (x % c + c) % c
        if let ExprKind::Binary(rem_op, rem_lhs, rem_rhs) = expr.kind
            && rem_op.node == BinOpKind::Rem
            && let ExprKind::Binary(add_op, add_lhs, add_rhs) = rem_lhs.kind
            && add_op.node == BinOpKind::Add
            && let ctxt = expr.span.ctxt()
            && rem_lhs.span.ctxt() == ctxt
            && rem_rhs.span.ctxt() == ctxt
            && add_lhs.span.ctxt() == ctxt
            && add_rhs.span.ctxt() == ctxt
            && !expr.span.in_external_macro(cx.sess().source_map())
            && let Some(const1) = check_for_unsigned_int_constant(cx, rem_rhs)
            && let Some((const2, add_other)) = check_for_either_unsigned_int_constant(cx, add_lhs, add_rhs)
            && let ExprKind::Binary(rem2_op, rem2_lhs, rem2_rhs) = add_other.kind
            && rem2_op.node == BinOpKind::Rem
            && const1 == const2
            && let Some(hir_id) = path_to_local(rem2_lhs)
            && let Some(const3) = check_for_unsigned_int_constant(cx, rem2_rhs)
            // Also ensures the const is nonzero since zero can't be a divisor
            && const2 == const3
            && rem2_lhs.span.ctxt() == ctxt
            && rem2_rhs.span.ctxt() == ctxt
            && self.msrv.meets(cx, msrvs::REM_EUCLID)
            && (self.msrv.meets(cx, msrvs::REM_EUCLID_CONST) || !is_in_const_context(cx))
        {
            // Apply only to params or locals with annotated types
            match cx.tcx.parent_hir_node(hir_id) {
                Node::Param(..) => (),
                Node::LetStmt(local) => {
                    let Some(ty) = local.ty else { return };
                    if matches!(ty.kind, TyKind::Infer(())) {
                        return;
                    }
                },
                _ => return,
            }

            let mut app = Applicability::MachineApplicable;
            let rem_of = snippet_with_context(cx, rem2_lhs.span, ctxt, "_", &mut app).0;
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
    let int_const = ConstEvalCtxt::new(cx).eval_full_int(expr)?;
    match int_const {
        FullInt::S(s) => s.try_into().ok(),
        FullInt::U(u) => Some(u),
    }
}
