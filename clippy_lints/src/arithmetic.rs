use rustc::hir;
use rustc::lint::*;
use syntax::codemap::Span;
use utils::span_lint;

/// **What it does:** This lint checks for plain integer arithmetic
///
/// **Why is this bad?** This is only checked against overflow in debug builds.
/// In some applications one wants explicitly checked, wrapping or saturating
/// arithmetic.
///
/// **Known problems:** None
///
/// **Example:**
/// ```
/// a + 1
/// ```
declare_restriction_lint! {
    pub INTEGER_ARITHMETIC,
    "Any integer arithmetic statement"
}

/// **What it does:** This lint checks for float arithmetic
///
/// **Why is this bad?** For some embedded systems or kernel development, it
/// can be useful to rule out floating-point numbers
///
/// **Known problems:** None
///
/// **Example:**
/// ```
/// a + 1.0
/// ```
declare_restriction_lint! {
    pub FLOAT_ARITHMETIC,
    "Any floating-point arithmetic statement"
}

#[derive(Copy, Clone, Default)]
pub struct Arithmetic {
    span: Option<Span>,
}

impl LintPass for Arithmetic {
    fn get_lints(&self) -> LintArray {
        lint_array!(INTEGER_ARITHMETIC, FLOAT_ARITHMETIC)
    }
}

impl LateLintPass for Arithmetic {
    fn check_expr(&mut self, cx: &LateContext, expr: &hir::Expr) {
        if let Some(_) = self.span {
            return;
        }
        match expr.node {
            hir::ExprBinary(ref op, ref l, ref r) => {
                match op.node {
                    hir::BiAnd | hir::BiOr | hir::BiBitAnd | hir::BiBitOr | hir::BiBitXor | hir::BiShl |
                    hir::BiShr | hir::BiEq | hir::BiLt | hir::BiLe | hir::BiNe | hir::BiGe | hir::BiGt => return,
                    _ => (),
                }
                let (l_ty, r_ty) = (cx.tcx.expr_ty(l), cx.tcx.expr_ty(r));
                if l_ty.is_integral() && r_ty.is_integral() {
                    span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                    self.span = Some(expr.span);
                } else if l_ty.is_floating_point() && r_ty.is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.span = Some(expr.span);
                }
            }
            hir::ExprUnary(hir::UnOp::UnNeg, ref arg) => {
                let ty = cx.tcx.expr_ty(arg);
                if ty.is_integral() {
                    span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                    self.span = Some(expr.span);
                } else if ty.is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.span = Some(expr.span);
                }
            }
            _ => (),
        }
    }

    fn check_expr_post(&mut self, _: &LateContext, expr: &hir::Expr) {
        if Some(expr.span) == self.span {
            self.span = None;
        }
    }
}
