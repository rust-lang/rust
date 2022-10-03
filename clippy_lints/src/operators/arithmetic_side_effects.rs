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
    "std::num::Wrapping",
    "std::string::String",
    "&str",
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

    /// Assuming that `expr` is a literal integer, checks operators (+=, -=, *, /) in a
    /// non-constant environment that won't overflow.
    fn has_valid_op(op: &Spanned<hir::BinOpKind>, expr: &hir::Expr<'_>) -> bool {
        if let hir::ExprKind::Lit(ref lit) = expr.kind &&
            let ast::LitKind::Int(value, _) = lit.node
        {
            match (&op.node, value) {
                (hir::BinOpKind::Add | hir::BinOpKind::Sub, 0) |
                (hir::BinOpKind::Mul, 0 | 1) => true,
                (hir::BinOpKind::Div | hir::BinOpKind::Rem, 0) => false,
                (hir::BinOpKind::Div | hir::BinOpKind::Rem, _) => true,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Checks if the given `expr` has any of the inner `allowed` elements.
    fn is_allowed_ty(&self, ty: Ty<'_>) -> bool {
        self.allowed
            .contains(ty.to_string().split('<').next().unwrap_or_default())
    }

    // For example, 8i32 or &i64::MAX.
    fn is_integral(ty: Ty<'_>) -> bool {
        ty.peel_refs().is_integral()
    }

    // Common entry-point to avoid code duplication.
    fn issue_lint(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        let msg = "arithmetic operation that can potentially result in unexpected side-effects";
        span_lint(cx, ARITHMETIC_SIDE_EFFECTS, expr.span, msg);
        self.expr_span = Some(expr.span);
    }

    /// * If `expr` is a literal integer like `1` or `i32::MAX`, returns itself.
    /// * Is `expr` is a literal integer reference like `&199`, returns the literal integer without
    ///   references.
    /// * If `expr` is anything else, returns `None`.
    fn literal_integer<'expr, 'tcx>(expr: &'expr hir::Expr<'tcx>) -> Option<&'expr hir::Expr<'tcx>> {
        if matches!(expr.kind, hir::ExprKind::Lit(_)) {
            return Some(expr);
        }
        if let hir::ExprKind::AddrOf(.., inn) = expr.kind && let hir::ExprKind::Lit(_) = inn.kind {
            return Some(inn)
        }
        None
    }

    /// Manages when the lint should be triggered. Operations in constant environments, hard coded
    /// types, custom allowed types and non-constant operations that won't overflow are ignored.
    fn manage_bin_ops<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &hir::Expr<'tcx>,
        op: &Spanned<hir::BinOpKind>,
        lhs: &hir::Expr<'tcx>,
        rhs: &hir::Expr<'tcx>,
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
        let lhs_ty = cx.typeck_results().expr_ty(lhs);
        let rhs_ty = cx.typeck_results().expr_ty(rhs);
        let lhs_and_rhs_have_the_same_ty = lhs_ty == rhs_ty;
        if lhs_and_rhs_have_the_same_ty && self.is_allowed_ty(lhs_ty) && self.is_allowed_ty(rhs_ty) {
            return;
        }
        let has_valid_op = if Self::is_integral(lhs_ty) && Self::is_integral(rhs_ty) {
            match (Self::literal_integer(lhs), Self::literal_integer(rhs)) {
                (None, None) => false,
                (None, Some(local_expr)) => Self::has_valid_op(op, local_expr),
                (Some(local_expr), None) => Self::has_valid_op(op, local_expr),
                (Some(_), Some(_)) => true,
            }
        } else {
            false
        };
        if !has_valid_op {
            self.issue_lint(cx, expr);
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for ArithmeticSideEffects {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &hir::Expr<'tcx>) {
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
