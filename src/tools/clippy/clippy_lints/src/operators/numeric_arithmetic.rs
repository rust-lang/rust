use clippy_utils::consts::constant_simple;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::is_integer_literal;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Span;

use super::{FLOAT_ARITHMETIC, INTEGER_ARITHMETIC};

#[derive(Default)]
pub struct Context {
    expr_id: Option<hir::HirId>,
    /// This field is used to check whether expressions are constants, such as in enum discriminants
    /// and consts
    const_span: Option<Span>,
}
impl Context {
    fn skip_expr(&mut self, e: &hir::Expr<'_>) -> bool {
        self.expr_id.is_some() || self.const_span.map_or(false, |span| span.contains(e.span))
    }

    pub fn check_binary<'tcx>(
        &mut self,
        cx: &LateContext<'tcx>,
        expr: &'tcx hir::Expr<'_>,
        op: hir::BinOpKind,
        l: &'tcx hir::Expr<'_>,
        r: &'tcx hir::Expr<'_>,
    ) {
        if self.skip_expr(expr) {
            return;
        }
        match op {
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
            match op {
                hir::BinOpKind::Div | hir::BinOpKind::Rem => match &r.kind {
                    hir::ExprKind::Lit(_lit) => (),
                    hir::ExprKind::Unary(hir::UnOp::Neg, expr) => {
                        if is_integer_literal(expr, 1) {
                            span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                            self.expr_id = Some(expr.hir_id);
                        }
                    },
                    _ => {
                        span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                        self.expr_id = Some(expr.hir_id);
                    },
                },
                _ => {
                    span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                    self.expr_id = Some(expr.hir_id);
                },
            }
        } else if r_ty.peel_refs().is_floating_point() && r_ty.peel_refs().is_floating_point() {
            span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
            self.expr_id = Some(expr.hir_id);
        }
    }

    pub fn check_negate<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, arg: &'tcx hir::Expr<'_>) {
        if self.skip_expr(expr) {
            return;
        }
        let ty = cx.typeck_results().expr_ty(arg);
        if constant_simple(cx, cx.typeck_results(), expr).is_none() {
            if ty.is_integral() {
                span_lint(cx, INTEGER_ARITHMETIC, expr.span, "integer arithmetic detected");
                self.expr_id = Some(expr.hir_id);
            } else if ty.is_floating_point() {
                span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
                self.expr_id = Some(expr.hir_id);
            }
        }
    }

    pub fn expr_post(&mut self, id: hir::HirId) {
        if Some(id) == self.expr_id {
            self.expr_id = None;
        }
    }

    pub fn enter_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_owner_def_id = cx.tcx.hir().body_owner_def_id(body.id());

        match cx.tcx.hir().body_owner_kind(body_owner_def_id) {
            hir::BodyOwnerKind::Static(_) | hir::BodyOwnerKind::Const => {
                let body_span = cx.tcx.hir().span_with_body(body_owner);

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

    pub fn body_post(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir().body_owner(body.id());
        let body_span = cx.tcx.hir().span_with_body(body_owner);

        if let Some(span) = self.const_span {
            if span.contains(body_span) {
                return;
            }
        }
        self.const_span = None;
    }
}
