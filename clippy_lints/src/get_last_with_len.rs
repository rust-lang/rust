//! lint on using `x.get(x.len() - 1)` instead of `x.last()`

use crate::utils::{match_type, paths, snippet_with_applicability, span_lint_and_sugg, SpanlessEq};
use if_chain::if_chain;
use rustc::hir::{BinOpKind, Expr, ExprKind};
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::LitKind;
use syntax::source_map::Spanned;
use syntax::symbol::Symbol;

declare_clippy_lint! {
    /// **What it does:** Checks for using `x.get(x.len() - 1)` instead of
    /// `x.last()`.
    ///
    /// **Why is this bad?** Using `x.last()` is easier to read and has the same
    /// result.
    ///
    /// Note that using `x[x.len() - 1]` is semantically different from
    /// `x.last()`.  Indexing into the array will panic on out-of-bounds
    /// accesses, while `x.get()` and `x.last()` will return `None`.
    ///
    /// There is another lint (get_unwrap) that covers the case of using
    /// `x.get(index).unwrap()` instead of `x[index]`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // Bad
    /// let x = vec![2, 3, 5];
    /// let last_element = x.get(x.len() - 1);
    ///
    /// // Good
    /// let x = vec![2, 3, 5];
    /// let last_element = x.last();
    /// ```
    pub GET_LAST_WITH_LEN,
    complexity,
    "Using `x.get(x.len() - 1)` when `x.last()` is correct and simpler"
}

declare_lint_pass!(GetLastWithLen => [GET_LAST_WITH_LEN]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for GetLastWithLen {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if_chain! {
            // Is a method call
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;

            // Method name is "get"
            if path.ident.name == Symbol::intern("get");

            // Argument 0 (the struct we're calling the method on) is a vector
            if let Some(struct_calling_on) = args.get(0);
            let struct_ty = cx.tables.expr_ty(struct_calling_on);
            if match_type(cx, struct_ty, &paths::VEC);

            // Argument to "get" is a subtraction
            if let Some(get_index_arg) = args.get(1);
            if let ExprKind::Binary(
                Spanned {
                    node: BinOpKind::Sub,
                    ..
                },
                lhs,
                rhs,
            ) = &get_index_arg.node;

            // LHS of subtraction is "x.len()"
            if let ExprKind::MethodCall(arg_lhs_path, _, lhs_args) = &lhs.node;
            if arg_lhs_path.ident.name == Symbol::intern("len");
            if let Some(arg_lhs_struct) = lhs_args.get(0);

            // The two vectors referenced (x in x.get(...) and in x.len())
            if SpanlessEq::new(cx).eq_expr(struct_calling_on, arg_lhs_struct);

            // RHS of subtraction is 1
            if let ExprKind::Lit(rhs_lit) = &rhs.node;
            if let LitKind::Int(rhs_value, ..) = rhs_lit.node;
            if rhs_value == 1;

            then {
                let mut applicability = Applicability::MachineApplicable;
                let vec_name = snippet_with_applicability(
                    cx,
                    struct_calling_on.span, "vec",
                    &mut applicability,
                );

                span_lint_and_sugg(
                    cx,
                    GET_LAST_WITH_LEN,
                    expr.span,
                    &format!("accessing last element with `{0}.get({0}.len() - 1)`", vec_name),
                    "try",
                    format!("{}.last()", vec_name),
                    applicability,
                );
            }
        }
    }
}
