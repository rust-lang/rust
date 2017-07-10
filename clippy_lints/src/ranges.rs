use rustc::lint::*;
use rustc::hir::*;
use utils::{is_integer_literal, paths, snippet, span_lint};
use utils::{higher, implements_trait, get_trait_def_id};

/// **What it does:** Checks for calling `.step_by(0)` on iterators,
/// which never terminates.
///
/// **Why is this bad?** This very much looks like an oversight, since with
/// `loop { .. }` there is an obvious better way to endlessly loop.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// for x in (5..5).step_by(0) { .. }
/// ```
declare_lint! {
    pub ITERATOR_STEP_BY_ZERO,
    Warn,
    "using `Iterator::step_by(0)`, which produces an infinite iterator"
}

/// **What it does:** Checks for zipping a collection with the range of `0.._.len()`.
///
/// **Why is this bad?** The code is better expressed with `.enumerate()`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// x.iter().zip(0..x.len())
/// ```
declare_lint! {
    pub RANGE_ZIP_WITH_LEN,
    Warn,
    "zipping iterator with a range when `enumerate()` would do"
}

#[derive(Copy,Clone)]
pub struct StepByZero;

impl LintPass for StepByZero {
    fn get_lints(&self) -> LintArray {
        lint_array!(ITERATOR_STEP_BY_ZERO, RANGE_ZIP_WITH_LEN)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for StepByZero {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprMethodCall(ref path, _, ref args) = expr.node {
            let name = path.name.as_str();

            // Range with step_by(0).
            if name == "step_by" && args.len() == 2 && has_step_by(cx, &args[0]) {
                use consts::{Constant, constant};
                use rustc_const_math::ConstInt::Usize;
                if let Some((Constant::Int(Usize(us)), _)) = constant(cx, &args[1]) {
                    if us.as_u64(cx.sess().target.uint_type) == 0 {
                        span_lint(cx,
                                  ITERATOR_STEP_BY_ZERO,
                                  expr.span,
                                  "Iterator::step_by(0) will panic at runtime");
                    }
                }
            } else if name == "zip" && args.len() == 2 {
                let iter = &args[0].node;
                let zip_arg = &args[1];
                if_let_chain! {[
                    // .iter() call
                    let ExprMethodCall(ref iter_path, _, ref iter_args ) = *iter,
                    iter_path.name == "iter",
                    // range expression in .zip() call: 0..x.len()
                    let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::range(zip_arg),
                    is_integer_literal(start, 0),
                    // .len() call
                    let ExprMethodCall(ref len_path, _, ref len_args) = end.node,
                    len_path.name == "len" && len_args.len() == 1,
                    // .iter() and .len() called on same Path
                    let ExprPath(QPath::Resolved(_, ref iter_path)) = iter_args[0].node,
                    let ExprPath(QPath::Resolved(_, ref len_path)) = len_args[0].node,
                    iter_path.segments == len_path.segments
                 ], {
                     span_lint(cx,
                               RANGE_ZIP_WITH_LEN,
                               expr.span,
                               &format!("It is more idiomatic to use {}.iter().enumerate()",
                                        snippet(cx, iter_args[0].span, "_")));
                }}
            }
        }
    }
}

fn has_step_by(cx: &LateContext, expr: &Expr) -> bool {
    // No need for walk_ptrs_ty here because step_by moves self, so it
    // can't be called on a borrowed range.
    let ty = cx.tables.expr_ty_adjusted(expr);

    get_trait_def_id(cx, &paths::ITERATOR).map_or(false, |iterator_trait| implements_trait(cx, ty, iterator_trait, &[]))
}
