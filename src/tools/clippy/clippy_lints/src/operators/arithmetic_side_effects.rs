#![allow(
    // False positive
    clippy::match_same_arms
)]

use super::ARITHMETIC_SIDE_EFFECTS;
use clippy_utils::{consts::constant_simple, diagnostics::span_lint};
use rustc_ast as ast;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::impl_lint_pass;
use rustc_span::source_map::{Span, Spanned};

const HARD_CODED_ALLOWED: &[&str] = &[
    "f32",
    "f64",
    "std::num::Saturating",
    "std::string::String",
    "std::num::Wrapping",
];

#[derive(Debug)]
pub struct ArithmeticSideEffects {
    allowed: FxHashSet<String>,
    // Used to check whether expressions are constants, such as in enum discriminants and consts
    const_span: Option<Span>,
    expr_span: Option<Span>,
}

impl_lint_pass!(ArithmeticSideEffects => [ARITHMETIC_SIDE_EFFECTS]);

impl ArithmeticSideEffects {
    #[must_use]
    pub fn new(mut allowed: FxHashSet<String>) -> Self {
        allowed.extend(HARD_CODED_ALLOWED.iter().copied().map(String::from));
        Self {
            allowed,
            const_span: None,
            expr_span: None,
        }
    }

    /// Checks assign operators (+=, -=, *=, /=) of integers in a non-constant environment that
    /// won't overflow.
    fn has_valid_assign_op(op: &Spanned<hir::BinOpKind>, rhs: &hir::Expr<'_>, rhs_refs: Ty<'_>) -> bool {
        if !Self::is_literal_integer(rhs, rhs_refs) {
            return false;
        }
        if let hir::BinOpKind::Div | hir::BinOpKind::Mul = op.node
            && let hir::ExprKind::Lit(ref lit) = rhs.kind
            && let ast::LitKind::Int(1, _) = lit.node
        {
            return true;
        }
        false
    }

    /// Checks "raw" binary operators (+, -, *, /) of integers in a non-constant environment
    /// already handled by the CTFE.
    fn has_valid_bin_op(lhs: &hir::Expr<'_>, lhs_refs: Ty<'_>, rhs: &hir::Expr<'_>, rhs_refs: Ty<'_>) -> bool {
        Self::is_literal_integer(lhs, lhs_refs) && Self::is_literal_integer(rhs, rhs_refs)
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

    /// Explicit integers like `1` or `i32::MAX`. Does not take into consideration references.
    fn is_literal_integer(expr: &hir::Expr<'_>, expr_refs: Ty<'_>) -> bool {
        let is_integral = expr_refs.is_integral();
        let is_literal = matches!(expr.kind, hir::ExprKind::Lit(_));
        is_integral && is_literal
    }

    fn issue_lint(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        span_lint(cx, ARITHMETIC_SIDE_EFFECTS, expr.span, "arithmetic detected");
        self.expr_span = Some(expr.span);
    }

    /// Manages when the lint should be triggered. Operations in constant environments, hard coded
    /// types, custom allowed types and non-constant operations that won't overflow are ignored.
    fn manage_bin_ops(
        &mut self,
        cx: &LateContext<'_>,
        expr: &hir::Expr<'_>,
        op: &Spanned<hir::BinOpKind>,
        lhs: &hir::Expr<'_>,
        rhs: &hir::Expr<'_>,
    ) {
        if constant_simple(cx, cx.typeck_results(), expr).is_some() {
            return;
        }
        if !matches!(
            op.node,
            hir::BinOpKind::Add
                | hir::BinOpKind::Sub
                | hir::BinOpKind::Mul
                | hir::BinOpKind::Div
                | hir::BinOpKind::Rem
                | hir::BinOpKind::Shl
                | hir::BinOpKind::Shr
        ) {
            return;
        };
        if self.is_allowed_ty(cx, lhs) || self.is_allowed_ty(cx, rhs) {
            return;
        }
        let lhs_refs = cx.typeck_results().expr_ty(lhs).peel_refs();
        let rhs_refs = cx.typeck_results().expr_ty(rhs).peel_refs();
        let has_valid_assign_op = Self::has_valid_assign_op(op, rhs, rhs_refs);
        if has_valid_assign_op || Self::has_valid_bin_op(lhs, lhs_refs, rhs, rhs_refs) {
            return;
        }
        self.issue_lint(cx, expr);
    }
}

impl<'tcx> LateLintPass<'tcx> for ArithmeticSideEffects {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if self.expr_span.is_some() || self.const_span.map_or(false, |sp| sp.contains(expr.span)) {
            return;
        }
        match &expr.kind {
            hir::ExprKind::Binary(op, lhs, rhs) | hir::ExprKind::AssignOp(op, lhs, rhs) => {
                self.manage_bin_ops(cx, expr, op, lhs, rhs);
            },
            hir::ExprKind::Unary(hir::UnOp::Neg, _) => {
                if constant_simple(cx, cx.typeck_results(), expr).is_none() {
                    self.issue_lint(cx, expr);
                }
            },
            _ => {},
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_owner_def_id = cx.tcx.hir().local_def_id(body_owner);
        let body_owner_kind = cx.tcx.hir().body_owner_kind(body_owner_def_id);
        if let hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) = body_owner_kind {
            let body_span = cx.tcx.hir().span_with_body(body_owner);
            if let Some(span) = self.const_span && span.contains(body_span) {
                return;
            }
            self.const_span = Some(body_span);
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
