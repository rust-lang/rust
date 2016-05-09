//! Checks for needless address of operations (`&`)
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc::hir::*;
use rustc::ty::TyRef;
use utils::{span_lint, in_macro};

/// **What it does:** This lint checks for address of operations (`&`) that are going to be dereferenced immediately by the compiler
///
/// **Why is this bad?** Suggests that the receiver of the expression borrows the expression
///
/// **Known problems:**
///
/// **Example:** `let x: &i32 = &&&&&&5;`
declare_lint! {
    pub NEEDLESS_BORROW,
    Warn,
    "taking a reference that is going to be automatically dereferenced"
}

#[derive(Copy,Clone)]
pub struct NeedlessBorrow;

impl LintPass for NeedlessBorrow {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_BORROW)
    }
}

impl LateLintPass for NeedlessBorrow {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if in_macro(cx, e.span) {
            return;
        }
        if let ExprAddrOf(MutImmutable, ref inner) = e.node {
            if let TyRef(..) = cx.tcx.expr_ty(inner).sty {
                let ty = cx.tcx.expr_ty(e);
                let adj_ty = cx.tcx.expr_ty_adjusted(e);
                if ty != adj_ty {
                    span_lint(cx,
                              NEEDLESS_BORROW,
                              e.span,
                              "this expression borrows a reference that is immediately dereferenced by the compiler");
                }
            }
        }
    }
}
