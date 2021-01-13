use crate::utils::span_lint;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Checks for needlessly including a base struct on update
    /// when all fields are changed anyway.
    ///
    /// This lint is not applied to structs marked with
    /// [non_exhaustive](https://doc.rust-lang.org/reference/attributes/type_system.html).
    ///
    /// **Why is this bad?** This will cost resources (because the base has to be
    /// somewhere), and make the code less readable.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # struct Point {
    /// #     x: i32,
    /// #     y: i32,
    /// #     z: i32,
    /// # }
    /// # let zero_point = Point { x: 0, y: 0, z: 0 };
    ///
    /// // Bad
    /// Point {
    ///     x: 1,
    ///     y: 1,
    ///     z: 1,
    ///     ..zero_point
    /// };
    ///
    /// // Ok
    /// Point {
    ///     x: 1,
    ///     y: 1,
    ///     ..zero_point
    /// };
    /// ```
    pub NEEDLESS_UPDATE,
    complexity,
    "using `Foo { ..base }` when there are no missing fields"
}

declare_lint_pass!(NeedlessUpdate => [NEEDLESS_UPDATE]);

impl<'tcx> LateLintPass<'tcx> for NeedlessUpdate {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Struct(_, ref fields, Some(ref base)) = expr.kind {
            let ty = cx.typeck_results().expr_ty(expr);
            if let ty::Adt(def, _) = ty.kind() {
                if fields.len() == def.non_enum_variant().fields.len()
                    && !def.variants[0_usize.into()].is_field_list_non_exhaustive()
                {
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
