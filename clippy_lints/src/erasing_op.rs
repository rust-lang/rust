use clippy_utils::consts::{constant_simple, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::same_type_and_consts;

use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TypeckResults;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for erasing operations, e.g., `x * 0`.
    ///
    /// ### Why is this bad?
    /// The whole expression can be replaced by zero.
    /// This is most likely not the intended outcome and should probably be
    /// corrected
    ///
    /// ### Example
    /// ```rust
    /// let x = 1;
    /// 0 / x;
    /// 0 * x;
    /// x & 0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ERASING_OP,
    correctness,
    "using erasing operations, e.g., `x * 0` or `y & 0`"
}

declare_lint_pass!(ErasingOp => [ERASING_OP]);

impl<'tcx> LateLintPass<'tcx> for ErasingOp {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }
        if let ExprKind::Binary(ref cmp, left, right) = e.kind {
            let tck = cx.typeck_results();
            match cmp.node {
                BinOpKind::Mul | BinOpKind::BitAnd => {
                    check(cx, tck, left, right, e);
                    check(cx, tck, right, left, e);
                },
                BinOpKind::Div => check(cx, tck, left, right, e),
                _ => (),
            }
        }
    }
}

fn different_types(tck: &TypeckResults<'_>, input: &Expr<'_>, output: &Expr<'_>) -> bool {
    let input_ty = tck.expr_ty(input).peel_refs();
    let output_ty = tck.expr_ty(output).peel_refs();
    !same_type_and_consts(input_ty, output_ty)
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    tck: &TypeckResults<'tcx>,
    op: &Expr<'tcx>,
    other: &Expr<'tcx>,
    parent: &Expr<'tcx>,
) {
    if constant_simple(cx, tck, op) == Some(Constant::Int(0)) {
        if different_types(tck, other, parent) {
            return;
        }
        span_lint(
            cx,
            ERASING_OP,
            parent.span,
            "this operation will always return zero. This is likely not the intended outcome",
        );
    }
}
