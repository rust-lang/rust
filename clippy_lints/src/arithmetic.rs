use crate::consts::constant_simple;
use crate::utils::span_lint;
use rustc::hir;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, impl_lint_pass};
use syntax::source_map::Span;

declare_clippy_lint! {
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
    /// # let a = 0;
    /// a + 1;
    /// ```
    pub INTEGER_ARITHMETIC,
    restriction,
    "any integer arithmetic statement"
}

declare_clippy_lint! {
    /// **What it does:** Checks for float arithmetic.
    ///
    /// **Why is this bad?** For some embedded systems or kernel development, it
    /// can be useful to rule out floating-point numbers.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # let a = 0.0;
    /// a + 1.0;
    /// ```
    pub FLOAT_ARITHMETIC,
    restriction,
    "any floating-point arithmetic statement"
}

#[derive(Copy, Clone, Default)]
pub struct Arithmetic {
    expr_span: Option<Span>,
    /// This field is used to check whether expressions are constants, such as in enum discriminants
    /// and consts
    const_span: Option<Span>,
}

impl_lint_pass!(Arithmetic => [INTEGER_ARITHMETIC, FLOAT_ARITHMETIC]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Arithmetic {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if self.expr_span.is_some() {
            return;
        }

        if let Some(span) = self.const_span {
            if span.contains(expr.span) {
                return;
            }
        }
        match &expr.node {
            hir::ExprKind::Binary(op, l, r) => {
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
                    self.expr_span = Some(expr.span);
                } else if l_ty.is_floating_point() && r_ty.is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.expr_span = Some(expr.span);
                }
            },
            hir::ExprKind::Unary(hir::UnOp::UnNeg, arg) => {
                let ty = cx.tables.expr_ty(arg);
                if ty.is_integral() {
                    if constant_simple(cx, cx.tables, expr).is_none() {
                        span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                        self.expr_span = Some(expr.span);
                    }
                } else if ty.is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.expr_span = Some(expr.span);
                }
            },
            _ => (),
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'a, 'tcx>, expr: &'tcx hir::Expr) {
        if Some(expr.span) == self.expr_span {
            self.expr_span = None;
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_, '_>, body: &hir::Body) {
        let body_owner = cx.tcx.hir().body_owner(body.id());

        match cx.tcx.hir().body_owner_kind(body_owner) {
            hir::BodyOwnerKind::Static(_) | hir::BodyOwnerKind::Const => {
                let body_span = cx.tcx.hir().span(body_owner);

                if let Some(span) = self.const_span {
                    if span.contains(body_span) {
                        return;
                    }
                }
                self.const_span = Some(body_span);
            },
            hir::BodyOwnerKind::Fn | hir::BodyOwnerKind::Closure => (),
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'_, '_>, body: &hir::Body) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_span = cx.tcx.hir().span(body_owner);

        if let Some(span) = self.const_span {
            if span.contains(body_span) {
                return;
            }
        }
        self.const_span = None;
    }
}
