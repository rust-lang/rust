use rustc::lint::*;
use syntax::ast::*;
use syntax::codemap::ExpnInfo;
use rustc::middle::ty::{TypeAndMut, TyRef};

use utils::{in_macro, span_lint};

declare_lint!(pub MUT_MUT, Warn,
              "usage of double-mut refs, e.g. `&mut &mut ...` (either copy'n'paste error, \
               or shows a fundamental misunderstanding of references)");

#[derive(Copy,Clone)]
pub struct MutMut;

impl LintPass for MutMut {
    fn get_lints(&self) -> LintArray {
        lint_array!(MUT_MUT)
    }

    fn check_expr(&mut self, cx: &Context, expr: &Expr) {
        cx.sess().codemap().with_expn_info(expr.span.expn_id,
            |info| check_expr_expd(cx, expr, info))
    }

    fn check_ty(&mut self, cx: &Context, ty: &Ty) {
        unwrap_mut(ty).and_then(unwrap_mut).map_or((), |_| span_lint(cx, MUT_MUT,
            ty.span, "generally you want to avoid `&mut &mut _` if possible"))
    }
}

fn check_expr_expd(cx: &Context, expr: &Expr, info: Option<&ExpnInfo>) {
    if in_macro(cx, info) { return; }

    fn unwrap_addr(expr : &Expr) -> Option<&Expr> {
        match expr.node {
            ExprAddrOf(MutMutable, ref e) => Option::Some(e),
            _ => Option::None
        }
    }

    unwrap_addr(expr).map_or((), |e| {
        unwrap_addr(e).map(|_| {
            span_lint(cx, MUT_MUT, expr.span,
                      "generally you want to avoid `&mut &mut _` if possible")
        }).unwrap_or_else(|| {
            if let TyRef(_, TypeAndMut{ty: _, mutbl: MutMutable}) =
                cx.tcx.expr_ty(e).sty {
                    span_lint(cx, MUT_MUT, expr.span,
                              "this expression mutably borrows a mutable reference. \
                               Consider reborrowing")
                }
        })
    })
}

fn unwrap_mut(ty : &Ty) -> Option<&Ty> {
    match ty.node {
        TyPtr(MutTy{ ty: ref pty, mutbl: MutMutable }) => Option::Some(pty),
        TyRptr(_, MutTy{ ty: ref pty, mutbl: MutMutable }) => Option::Some(pty),
        _ => Option::None
    }
}
