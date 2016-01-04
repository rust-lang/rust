use rustc::lint::*;
use rustc::middle::const_eval::EvalHint::ExprTypeChecked;
use rustc::middle::const_eval::{eval_const_expr_partial, ConstVal};
use rustc::middle::ty::TyArray;
use rustc_front::hir::*;
use utils::span_lint;

/// **What it does:** Check for out of bounds array indexing with a constant index.
///
/// **Why is this bad?** This will always panic at runtime.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
///
/// ```
/// let x = [1,2,3,4];
/// ...
/// x[9];
/// ```
declare_lint! {
    pub OUT_OF_BOUNDS_INDEXING,
    Deny,
    "out of bound constant indexing"
}

#[derive(Copy,Clone)]
pub struct ArrayIndexing;

impl LintPass for ArrayIndexing {
    fn get_lints(&self) -> LintArray {
        lint_array!(OUT_OF_BOUNDS_INDEXING)
    }
}

impl LateLintPass for ArrayIndexing {
    fn check_expr(&mut self, cx: &LateContext, e: &Expr) {
        if let ExprIndex(ref array, ref index) = e.node {
            let ty = cx.tcx.expr_ty(array);

            if let TyArray(_, size) = ty.sty {
                let index = eval_const_expr_partial(cx.tcx, &index, ExprTypeChecked, None);
                if let Ok(ConstVal::Uint(index)) = index {
                    if size as u64 <= index {
                        span_lint(cx, OUT_OF_BOUNDS_INDEXING, e.span, "const index-expr is out of bounds");
                    }
                }
            }
        }
    }
}
