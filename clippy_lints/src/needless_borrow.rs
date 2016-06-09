//! Checks for needless address of operations (`&`)
//!
//! This lint is **warn** by default

use rustc::lint::*;
use rustc::hir::{ExprAddrOf, Expr, MutImmutable};
use rustc::ty::TyRef;
use utils::{span_lint, in_macro};
use rustc::ty::adjustment::AutoAdjustment::AdjustDerefRef;

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
                if let Some(&AdjustDerefRef(ref deref)) = cx.tcx.tables.borrow().adjustments.get(&e.id) {
                    if deref.autoderefs > 1 && deref.autoref.is_some() {
                        span_lint(cx,
                                  NEEDLESS_BORROW,
                                  e.span,
                                  "this expression borrows a reference that is immediately dereferenced by the \
                                   compiler");
                    }
                }
            }
        }
    }
}
