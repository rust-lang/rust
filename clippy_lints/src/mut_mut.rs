use rustc::lint::*;
use rustc::ty::{TypeAndMut, TyRef};
use rustc::hir::*;
use utils::{in_external_macro, span_lint};

/// **What it does:** This lint checks for instances of `mut mut` references.
///
/// **Why is this bad?** Multiple `mut`s don't add anything meaningful to the source.
///
/// **Known problems:** None
///
/// **Example:** `let x = &mut &mut y;`
declare_lint! {
    pub MUT_MUT,
    Allow,
    "usage of double-mut refs, e.g. `&mut &mut ...` (either copy'n'paste error, \
     or shows a fundamental misunderstanding of references)"
}

#[derive(Copy,Clone)]
pub struct MutMut;

impl LintPass for MutMut {
    fn get_lints(&self) -> LintArray {
        lint_array!(MUT_MUT)
    }
}

impl LateLintPass for MutMut {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if in_external_macro(cx, expr.span) {
            return;
        }

        if let ExprAddrOf(MutMutable, ref e) = expr.node {
            if let ExprAddrOf(MutMutable, _) = e.node {
                span_lint(cx, MUT_MUT, expr.span, "generally you want to avoid `&mut &mut _` if possible");
            } else {
                if let TyRef(_, TypeAndMut { mutbl: MutMutable, .. }) = cx.tcx.expr_ty(e).sty {
                    span_lint(cx,
                              MUT_MUT,
                              expr.span,
                              "this expression mutably borrows a mutable reference. Consider reborrowing");
                }
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext, ty: &Ty) {
        if let TyRptr(_, MutTy { ty: ref pty, mutbl: MutMutable }) = ty.node {
            if let TyRptr(_, MutTy { mutbl: MutMutable, .. }) = pty.node {
                span_lint(cx, MUT_MUT, ty.span, "generally you want to avoid `&mut &mut _` if possible");
            }
        }
    }
}
