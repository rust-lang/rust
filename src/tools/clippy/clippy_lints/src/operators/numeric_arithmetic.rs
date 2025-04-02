use super::FLOAT_ARITHMETIC;
use clippy_utils::consts::ConstEvalCtxt;
use clippy_utils::diagnostics::span_lint;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::Span;

#[derive(Default)]
pub struct Context {
    expr_id: Option<hir::HirId>,
    /// This field is used to check whether expressions are constants, such as in enum discriminants
    /// and consts
    const_span: Option<Span>,
}
impl Context {
    fn skip_expr(&mut self, e: &hir::Expr<'_>) -> bool {
        self.expr_id.is_some() || self.const_span.is_some_and(|span| span.contains(e.span))
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
        if l_ty.peel_refs().is_floating_point() && r_ty.peel_refs().is_floating_point() {
            span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
            self.expr_id = Some(expr.hir_id);
        }
    }

    pub fn check_negate<'tcx>(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>, arg: &'tcx hir::Expr<'_>) {
        if self.skip_expr(expr) {
            return;
        }
        let ty = cx.typeck_results().expr_ty(arg);
        if ConstEvalCtxt::new(cx).eval_simple(expr).is_none() && ty.is_floating_point() {
            span_lint(cx, FLOAT_ARITHMETIC, expr.span, "floating-point arithmetic detected");
            self.expr_id = Some(expr.hir_id);
        }
    }

    pub fn expr_post(&mut self, id: hir::HirId) {
        if Some(id) == self.expr_id {
            self.expr_id = None;
        }
    }

    pub fn enter_body(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir_body_owner(body.id());
        let body_owner_def_id = cx.tcx.hir_body_owner_def_id(body.id());

        match cx.tcx.hir_body_owner_kind(body_owner_def_id) {
            hir::BodyOwnerKind::Static(_) | hir::BodyOwnerKind::Const { .. } => {
                let body_span = cx.tcx.hir_span_with_body(body_owner);

                if let Some(span) = self.const_span {
                    if span.contains(body_span) {
                        return;
                    }
                }
                self.const_span = Some(body_span);
            },
            hir::BodyOwnerKind::Fn | hir::BodyOwnerKind::Closure | hir::BodyOwnerKind::GlobalAsm => (),
        }
    }

    pub fn body_post(&mut self, cx: &LateContext<'_>, body: &hir::Body<'_>) {
        let body_owner = cx.tcx.hir_body_owner(body.id());
        let body_span = cx.tcx.hir_span_with_body(body_owner);

        if let Some(span) = self.const_span {
            if span.contains(body_span) {
                return;
            }
        }
        self.const_span = None;
    }
}
