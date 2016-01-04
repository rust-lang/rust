use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::middle::ty::TyStruct;
use rustc_front::hir::{Expr, ExprStruct};

use utils::span_lint;

/// **What it does:** This lint `Warn`s on needlessly including a base struct on update when all fields are changed anyway.
///
/// **Why is this bad?** This will cost resources (because the base has to be somewhere), and make the code less readable.
///
/// **Known problems:** None.
///
/// **Example:** `Point { x: 1, y: 0, ..zero_point }`
declare_lint! {
    pub NEEDLESS_UPDATE,
    Warn,
    "using `{ ..base }` when there are no missing fields"
}

#[derive(Copy, Clone)]
pub struct NeedlessUpdatePass;

impl LintPass for NeedlessUpdatePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_UPDATE)
    }
}

impl LateLintPass for NeedlessUpdatePass {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if let ExprStruct(_, ref fields, Some(ref base)) = expr.node {
            let ty = cx.tcx.expr_ty(expr);
            if let TyStruct(def, _) = ty.sty {
                if fields.len() == def.struct_variant().fields.len() {
                    span_lint(cx,
                              NEEDLESS_UPDATE,
                              base.span,
                              "struct update has no effect, all the fields in the struct have already been specified");
                }
            }
        }
    }
}
