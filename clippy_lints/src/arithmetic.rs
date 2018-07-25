use crate::utils::span_lint;
use rustc::hir;
use rustc::lint::*;
use rustc::{declare_lint, lint_array};
use syntax::codemap::Span;

/// **What it does:** Checks for plain integer arithmetic.
///
/// **Why is this bad?** This is only checked against overflow in debug builds.
/// In some applications one wants explicitly checked, wrapping or saturating
/// arithmetic.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// a + 1
/// ```
declare_clippy_lint! {
    pub INTEGER_ARITHMETIC,
    restriction,
    "any integer arithmetic statement"
}

/// **What it does:** Checks for float arithmetic.
///
/// **Why is this bad?** For some embedded systems or kernel development, it
/// can be useful to rule out floating-point numbers.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// a + 1.0
/// ```
declare_clippy_lint! {
    pub FLOAT_ARITHMETIC,
    restriction,
    "any floating-point arithmetic statement"
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Arithmetic {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if self.span.is_some() {
            return;
        }
        match expr.node {
            hir::ExprKind::Binary(ref op, ref l, ref r) => {
                match op.node {
                    hir::BinOpKind::And
                    | hir::BinOpKind::Or
                    | hir::BinOpKind::BitAnd
                    | hir::BinOpKind::BitOr
                    | hir::BinOpKind::BitXor
                    | hir::BinOpKind::Shl
                    | hir::BinOpKind::Shr
                    | hir::BinOpKind::Eq
                    | hir::BinOpKind::Lt
                    | hir::BinOpKind::Le
                    | hir::BinOpKind::Ne
                    | hir::BinOpKind::Ge
                    | hir::BinOpKind::Gt => return,
                    _ => (),
                }
                let (l_ty, r_ty) = (cx.tables.expr_ty(l), cx.tables.expr_ty(r));
                if l_ty.is_integral() && r_ty.is_integral() {
                    span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                    self.span = Some(expr.span);
                } else if l_ty.is_floating_point() && r_ty.is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.span = Some(expr.span);
                }
            },
            hir::ExprKind::Unary(hir::UnOp::UnNeg, ref arg) => {
                let ty = cx.tables.expr_ty(arg);
                if ty.is_integral() {
                    span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                    self.span = Some(expr.span);
                } else if ty.is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.span = Some(expr.span);
                }
            },
            _ => (),
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if Some(expr.span) == self.span {
            self.span = None;
        }
    }
}
