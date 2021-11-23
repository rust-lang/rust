use clippy_utils::consts::constant_simple;
use clippy_utils::diagnostics::span_lint;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for integer arithmetic operations which could overflow or panic.
    ///
    /// Specifically, checks for any operators (`+`, `-`, `*`, `<<`, etc) which are capable
    /// of overflowing according to the [Rust
    /// Reference](https://doc.rust-lang.org/reference/expressions/operator-expr.html#overflow),
    /// or which can panic (`/`, `%`). No bounds analysis or sophisticated reasoning is
    /// attempted.
    ///
    /// ### Why is this bad?
    /// Integer overflow will trigger a panic in debug builds or will wrap in
    /// release mode. Division by zero will cause a panic in either mode. In some applications one
    /// wants explicitly checked, wrapping or saturating arithmetic.
    ///
    /// ### Example
    /// ```rust
    /// # let a = 0;
    /// a + 1;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub INTEGER_ARITHMETIC,
    restriction,
    "any integer arithmetic expression which could overflow or panic"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for float arithmetic.
    ///
    /// ### Why is this bad?
    /// For some embedded systems or kernel development, it
    /// can be useful to rule out floating-point numbers.
    ///
    /// ### Example
    /// ```rust
    /// # let a = 0.0;
    /// a + 1.0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
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

impl<'tcx> LateLintPass<'tcx> for Arithmetic {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if self.expr_span.is_some() {
            return;
        }

        if let Some(span) = self.const_span {
            if span.contains(expr.span) {
                return;
            }
        }
        match &expr.kind {
            hir::ExprKind::Binary(op, l, r) | hir::ExprKind::AssignOp(op, l, r) => {
                match op.node {
                    hir::BinOpKind::And
                    | hir::BinOpKind::Or
                    | hir::BinOpKind::BitAnd
                    | hir::BinOpKind::BitOr
                    | hir::BinOpKind::BitXor
                    | hir::BinOpKind::Eq
                    | hir::BinOpKind::Lt
                    | hir::BinOpKind::Le
                    | hir::BinOpKind::Ne
                    | hir::BinOpKind::Ge
                    | hir::BinOpKind::Gt => return,
                    _ => (),
                }

                let (l_ty, r_ty) = (cx.typeck_results().expr_ty(l), cx.typeck_results().expr_ty(r));
                if l_ty.peel_refs().is_integral() && r_ty.peel_refs().is_integral() {
                    match op.node {
                        hir::BinOpKind::Div | hir::BinOpKind::Rem => match &r.kind {
                            hir::ExprKind::Lit(_lit) => (),
                            hir::ExprKind::Unary(hir::UnOp::Neg, expr) => {
                                if let hir::ExprKind::Lit(lit) = &expr.kind {
                                    if let rustc_ast::ast::LitKind::Int(1, _) = lit.node {
                                        span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                                        self.expr_span = Some(expr.span);
                                    }
                                }
                            },
                            _ => {
                                span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                                self.expr_span = Some(expr.span);
                            },
                        },
                        _ => {
                            span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                            self.expr_span = Some(expr.span);
                        },
                    }
                } else if r_ty.peel_refs().is_floating_point() && r_ty.peel_refs().is_floating_point() {
                    span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                    self.expr_span = Some(expr.span);
                }
            },
            hir::ExprKind::Unary(hir::UnOp::Neg, arg) => {
                let ty = cx.typeck_results().expr_ty(arg);
                if constant_simple(cx, cx.typeck_results(), expr).is_none() {
                    if ty.is_integral() {
                        span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                        self.expr_span = Some(expr.span);
                    } else if ty.is_floating_point() {
                        span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                        self.expr_span = Some(expr.span);
                    }
                }
            },
            _ => (),
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if Some(expr.span) == self.expr_span {
            self.expr_span = None;
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
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

    fn check_body_post(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
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
