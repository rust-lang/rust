use rustc::lint::{Context, LintArray, LintPass};
use rustc::middle::ty::TypeVariants::TyStruct;
use syntax::ast::*;
use syntax::codemap::Spanned;
use utils::{match_def_path};

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
    if let TyStruct(did, _) = cx.tcx.expr_ty(expr).sty {
        // Note: RangeTo and RangeFull don't have step_by
        match_def_path(cx, did.did, &["core", "ops", "Range"]) ||
        match_def_path(cx, did.did, &["core", "ops", "RangeFrom"])
    } else { false }
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
