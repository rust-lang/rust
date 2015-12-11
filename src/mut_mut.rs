use rustc::lint::*;
use rustc_front::hir::*;
use rustc::middle::ty::{TypeAndMut, TyRef};

use utils::{in_external_macro, span_lint};

/// **What it does:** This lint checks for instances of `mut mut` references. It is `Warn` by default.
///
/// **Why is this bad?** Multiple `mut`s don't add anything meaningful to the source.
///
/// **Known problems:** None
///
/// **Example:** `let x = &mut &mut y;`
declare_lint!(pub MUT_MUT, Allow,
              "usage of double-mut refs, e.g. `&mut &mut ...` (either copy'n'paste error, \
               or shows a fundamental misunderstanding of references)");

#[derive(Copy,Clone)]
pub struct MutMut;

impl LintPass for MutMut {
    fn get_lints(&self) -> LintArray {
        lint_array!(MUT_MUT)
    }
}

impl LateLintPass for MutMut {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        check_expr_mut(cx, expr)
    }

    fn check_ty(&mut self, cx: &LateContext, ty: &Ty) {
        unwrap_mut(ty).and_then(unwrap_mut).map_or((), |_| span_lint(cx, MUT_MUT,
            ty.span, "generally you want to avoid `&mut &mut _` if possible"))
    }
}

fn check_expr_mut(cx: &LateContext, expr: &Expr) {
    if in_external_macro(cx, expr.span) { return; }

    fn unwrap_addr(expr: &Expr) -> Option<&Expr> {
        match expr.node {
            ExprAddrOf(MutMutable, ref e) => Some(e),
            _ => None
        }
    }

    unwrap_addr(expr).map_or((), |e| {
        unwrap_addr(e).map_or_else(
            || {
                if let TyRef(_, TypeAndMut{mutbl: MutMutable, ..}) =
                    cx.tcx.expr_ty(e).sty {
                        span_lint(cx, MUT_MUT, expr.span,
                                  "this expression mutably borrows a mutable reference. \
                                   Consider reborrowing")
                }
            },
            |_| {
                span_lint(cx, MUT_MUT, expr.span,
                          "generally you want to avoid `&mut &mut _` if possible")
            }
        )
    })
}

fn unwrap_mut(ty: &Ty) -> Option<&Ty> {
    match ty.node {
        TyRptr(_, MutTy{ ty: ref pty, mutbl: MutMutable }) => Some(pty),
        _ => None
    }
}
