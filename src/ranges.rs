use rustc::lint::{Context, LintArray, LintPass};
use syntax::ast::*;
use syntax::codemap::Spanned;
use utils::match_type;

declare_lint! {
    pub RANGE_STEP_BY_ZERO, Warn,
    "using Range::step_by(0), which produces an infinite iterator"
}

#[derive(Copy,Clone)]
pub struct StepByZero;

impl LintPass for StepByZero {
    fn get_lints(&self) -> LintArray {
        lint_array!(RANGE_STEP_BY_ZERO)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        if let ExprMethodCall(Spanned { node: ref ident, .. }, _,
                              ref args) = expr.node {
            // Only warn on literal ranges.
            if ident.name.as_str() == "step_by" && args.len() == 2 &&
                is_range(cx, &args[0]) && is_lit_zero(&args[1]) {
                cx.span_lint(RANGE_STEP_BY_ZERO, expr.span,
                             "Range::step_by(0) produces an infinite iterator. \
                              Consider using `std::iter::repeat()` instead")
            }
        }
    }
}

fn is_range(cx: &Context, expr: &Expr) -> bool {
    // No need for walk_ptrs_ty here because step_by moves self, so it
    // can't be called on a borrowed range.
    let ty = cx.tcx.expr_ty(expr);
    // Note: RangeTo and RangeFull don't have step_by
    match_type(cx, ty, &["core", "ops", "Range"]) || match_type(cx, ty, &["core", "ops", "RangeFrom"])
}

fn is_lit_zero(expr: &Expr) -> bool {
    // FIXME: use constant folding
    if let ExprLit(ref spanned) = expr.node {
        if let LitInt(0, _) = spanned.node {
            return true;
        }
    }
    false
}
