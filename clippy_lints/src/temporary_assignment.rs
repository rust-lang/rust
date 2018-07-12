use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::hir::{Expr, ExprKind};
use crate::utils::is_adjusted;
use crate::utils::span_lint;

/// **What it does:** Checks for construction of a structure or tuple just to
/// assign a value in it.
///
/// **Why is this bad?** Readability. If the structure is only created to be
/// updated, why not write the structure you want in the first place?
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// (0, 0).0 = 1
/// ```
declare_clippy_lint! {
    pub TEMPORARY_ASSIGNMENT,
    complexity,
    "assignments to temporaries"
}

fn is_temporary(expr: &Expr) -> bool {
    match expr.node {
        ExprKind::Struct(..) | ExprKind::Tup(..) => true,
        _ => false,
    }
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TEMPORARY_ASSIGNMENT)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Assign(ref target, _) = expr.node {
            if let ExprKind::Field(ref base, _) = target.node {
                if is_temporary(base) && !is_adjusted(cx, base) {
                    span_lint(cx, TEMPORARY_ASSIGNMENT, expr.span, "assignment to temporary");
                }
            }
        }
    }
}
