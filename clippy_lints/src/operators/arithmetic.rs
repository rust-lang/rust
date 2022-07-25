#![allow(
    // False positive
    clippy::match_same_arms
)]

use super::ARITHMETIC;
use clippy_utils::{consts::constant_simple, diagnostics::span_lint};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::source_map::Span;

const HARD_CODED_ALLOWED: &[&str] = &["std::num::Saturating", "std::string::String", "std::num::Wrapping"];

#[derive(Debug)]
pub struct Arithmetic {
    allowed: FxHashSet<String>,
    // Used to check whether expressions are constants, such as in enum discriminants and consts
    const_span: Option<Span>,
    expr_span: Option<Span>,
}

impl_lint_pass!(Arithmetic => [ARITHMETIC]);

impl Arithmetic {
    #[must_use]
    pub fn new(mut allowed: FxHashSet<String>) -> Self {
        allowed.extend(HARD_CODED_ALLOWED.iter().copied().map(String::from));
        Self {
            allowed,
            const_span: None,
            expr_span: None,
        }
    }

    /// Checks if the given `expr` has any of the inner `allowed` elements.
    fn is_allowed_ty(&self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
        self.allowed.contains(
            cx.typeck_results()
                .expr_ty(expr)
                .to_string()
                .split('<')
                .next()
                .unwrap_or_default(),
        )
    }

    fn issue_lint(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        span_lint(cx, ARITHMETIC, expr.span, "arithmetic detected");
        self.expr_span = Some(expr.span);
    }
}

impl<'tcx> LateLintPass<'tcx> for Arithmetic {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if self.expr_span.is_some() {
            return;
        }
        if let Some(span) = self.const_span && span.contains(expr.span) {
            return;
        }
        match &expr.kind {
            hir::ExprKind::Binary(op, lhs, rhs) | hir::ExprKind::AssignOp(op, lhs, rhs) => {
                let (
                    hir::BinOpKind::Add
                    | hir::BinOpKind::Sub
                    | hir::BinOpKind::Mul
                    | hir::BinOpKind::Div
                    | hir::BinOpKind::Rem
                    | hir::BinOpKind::Shl
                    | hir::BinOpKind::Shr
                ) = op.node else {
                    return;
                };
                if self.is_allowed_ty(cx, lhs) || self.is_allowed_ty(cx, rhs) {
                    return;
                }
                self.issue_lint(cx, expr);
            },
            hir::ExprKind::Unary(hir::UnOp::Neg, _) => {
                // CTFE already takes care of things like `-1` that do not overflow.
                if constant_simple(cx, cx.typeck_results(), expr).is_none() {
                    self.issue_lint(cx, expr);
                }
            },
            _ => {},
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner_def_id(body.id());
        match cx.tcx.hir().body_owner_kind(body_owner) {
            hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => {
                let body_span = cx.tcx.def_span(body_owner);
                if let Some(span) = self.const_span && span.contains(body_span) {
                    return;
                }
                self.const_span = Some(body_span);
            },
            hir::BodyOwnerKind::Closure | hir::BodyOwnerKind::Fn => {},
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_span = cx.tcx.hir().span(body_owner);
        if let Some(span) = self.const_span && span.contains(body_span) {
            return;
        }
        self.const_span = None;
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if Some(expr.span) == self.expr_span {
            self.expr_span = None;
        }
    }
}
