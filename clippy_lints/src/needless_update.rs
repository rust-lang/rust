use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::hir::{Expr, ExprStruct};
use utils::span_lint;

/// **What it does:** Checks for needlessly including a base struct on update
/// when all fields are changed anyway.
///
/// **Why is this bad?** This will cost resources (because the base has to be
/// somewhere), and make the code less readable.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// Point { x: 1, y: 0, ..zero_point }
/// ```
declare_lint! {
    pub NEEDLESS_UPDATE,
    Warn,
    "using `Foo { ..base }` when there are no missing fields"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_UPDATE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprStruct(_, ref fields, Some(ref base)) = expr.node {
            let ty = cx.tables.expr_ty(expr);
            if let ty::TyAdt(def, _) = ty.sty {
                if fields.len() == def.struct_variant().fields.len() {
                    span_lint(
                        cx,
                        NEEDLESS_UPDATE,
                        base.span,
                        "struct update has no effect, all the fields in the struct have already been specified",
                    );
                }
            }
        }
    }
}
