use crate::utils::span_lint;
use rustc::hir::{Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use rustc::{declare_tool_lint, lint_array};

declare_clippy_lint! {
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
    /// Point {
    ///     x: 1,
    ///     y: 0,
    ///     ..zero_point
    /// }
    /// ```
    pub NEEDLESS_UPDATE,
    complexity,
    "using `Foo { ..base }` when there are no missing fields"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_UPDATE)
    }

    fn name(&self) -> &'static str {
        "NeedUpdate"
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Struct(_, ref fields, Some(ref base)) = expr.node {
            let ty = cx.tables.expr_ty(expr);
            if let ty::Adt(def, _) = ty.sty {
                if fields.len() == def.non_enum_variant().fields.len() {
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
