use rustc::lint::*;
use rustc::hir::*;
use utils::{is_direct_expn_of, implements_trait, span_lint};

/// **What it does:** Checks for `assert!(x == y)` which can be better written
/// as `assert_eq!(x, y)` if `x` and `y` implement `Debug` trait.
///
/// **Why is this bad?** `assert_eq` provides better assertion failure reporting.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
/// ```rust
/// let (x, y) = (1, 2);
///
/// assert!(x == y);  // assertion failed: x == y
/// assert_eq!(x, y); // assertion failed: `(left == right)` (left: `1`, right: `2`)
/// ```
declare_lint! {
    pub SHOULD_ASSERT_EQ,
    Warn,
    "using `assert` macro for asserting equality"
}

pub struct ShouldAssertEq;

impl LintPass for ShouldAssertEq {
    fn get_lints(&self) -> LintArray {
        lint_array![SHOULD_ASSERT_EQ]
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for ShouldAssertEq {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if_let_chain! {[
            let ExprIf(ref cond, ..) = e.node,
            let ExprUnary(UnOp::UnNot, ref cond) = cond.node,
            let ExprBinary(ref binop, ref expr1, ref expr2) = cond.node,
            binop.node == BinOp_::BiEq,
            is_direct_expn_of(cx, e.span, "assert").is_some(),
            let Some(debug_trait) = cx.tcx.lang_items.debug_trait(),
        ], {
            let ty1 = cx.tables.expr_ty(expr1);
            let ty2 = cx.tables.expr_ty(expr2);

            let parent = cx.tcx.hir.get_parent(e.id);

            if implements_trait(cx, ty1, debug_trait, &[], Some(parent)) &&
                implements_trait(cx, ty2, debug_trait, &[], Some(parent)) {
                span_lint(cx, SHOULD_ASSERT_EQ, e.span, "use `assert_eq` for better reporting");
            }
        }}
    }
}
