use rustc::lint::*;
use rustc_front::hir::*;
use utils::{get_parent_expr, in_macro, is_block_equal, is_exp_equal, span_lint, span_note_and_lint};

/// **What it does:** This lint checks for consecutive `ifs` with the same condition. This lint is
/// `Warn` by default.
///
/// **Why is this bad?** This is probably a copy & paste error.
///
/// **Known problems:** Hopefully none.
///
/// **Example:** `if a == b { .. } else if a == b { .. }`
declare_lint! {
    pub IFS_SAME_COND,
    Warn,
    "consecutive `ifs` with the same condition"
}

/// **What it does:** This lint checks for `if/else` with the same body as the *then* part and the
/// *else* part. This lint is `Warn` by default.
///
/// **Why is this bad?** This is probably a copy & paste error.
///
/// **Known problems:** Hopefully none.
///
/// **Example:** `if .. { 42 } else { 42 }`
declare_lint! {
    pub IF_SAME_THEN_ELSE,
    Warn,
    "if with the same *then* and *else* blocks"
}

#[derive(Copy, Clone, Debug)]
pub struct CopyAndPaste;

impl LintPass for CopyAndPaste {
    fn get_lints(&self) -> LintArray {
        lint_array![
            IFS_SAME_COND,
            IF_SAME_THEN_ELSE
        ]
    }
}

impl LateLintPass for CopyAndPaste {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if !in_macro(cx, expr.span) {
            lint_same_then_else(cx, expr);
            lint_same_cond(cx, expr);
        }
    }
}

/// Implementation of `IF_SAME_THEN_ELSE`.
fn lint_same_then_else(cx: &LateContext, expr: &Expr) {
    if let ExprIf(_, ref then_block, Some(ref else_expr)) = expr.node {
        let must_lint = if let ExprBlock(ref else_block) = else_expr.node {
            is_block_equal(cx, &then_block, &else_block, false)
        }
        else {
            false
        };

        if must_lint {
            span_lint(cx, IF_SAME_THEN_ELSE, expr.span, "this if has the same then and else blocks");
        }
    }
}

/// Implementation of `IFS_SAME_COND`.
fn lint_same_cond(cx: &LateContext, expr: &Expr) {
    // skip ifs directly in else, it will be checked in the parent if
    if let Some(&Expr{node: ExprIf(_, _, Some(ref else_expr)), ..}) = get_parent_expr(cx, expr) {
        if else_expr.id == expr.id {
            return;
        }
    }

    let conds = condition_sequence(expr);

    for (n, i) in conds.iter().enumerate() {
        for j in conds.iter().skip(n+1) {
            if is_exp_equal(cx, i, j, true) {
                span_note_and_lint(cx, IFS_SAME_COND, j.span, "this if has the same condition as a previous if", i.span, "same as this");
            }
        }
    }
}

/// Return the list of conditions expression in a sequence of `if/else`.
/// Eg. would return `[a, b]` for the expression `if a {..} else if b {..}`.
fn condition_sequence(mut expr: &Expr) -> Vec<&Expr> {
    let mut result = vec![];

    while let ExprIf(ref cond, _, ref else_expr) = expr.node {
        result.push(&**cond);

        if let Some(ref else_expr) = *else_expr {
            expr = else_expr;
        }
        else {
            break;
        }
    }

    result
}
