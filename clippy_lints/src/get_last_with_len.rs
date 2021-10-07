//! lint on using `x.get(x.len() - 1)` instead of `x.last()`

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::SpanlessEq;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for using `x.get(x.len() - 1)` instead of
    /// `x.last()`.
    ///
    /// ### Why is this bad?
    /// Using `x.last()` is easier to read and has the same
    /// result.
    ///
    /// Note that using `x[x.len() - 1]` is semantically different from
    /// `x.last()`.  Indexing into the array will panic on out-of-bounds
    /// accesses, while `x.get()` and `x.last()` will return `None`.
    ///
    /// There is another lint (get_unwrap) that covers the case of using
    /// `x.get(index).unwrap()` instead of `x[index]`.
    ///
    /// ### Example
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

impl<'tcx> LateLintPass<'tcx> for GetLastWithLen {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            // Is a method call
            if let ExprKind::MethodCall(path, _, args, _) = expr.kind;

            // Method name is "get"
            if path.ident.name == sym!(get);

            // Argument 0 (the struct we're calling the method on) is a vector
            if let Some(struct_calling_on) = args.get(0);
            let struct_ty = cx.typeck_results().expr_ty(struct_calling_on);
            if is_type_diagnostic_item(cx, struct_ty, sym::Vec);

            // Argument to "get" is a subtraction
            if let Some(get_index_arg) = args.get(1);
            if let ExprKind::Binary(
                Spanned {
                    node: BinOpKind::Sub,
                    ..
                },
                lhs,
                rhs,
            ) = &get_index_arg.kind;

            // LHS of subtraction is "x.len()"
            if let ExprKind::MethodCall(arg_lhs_path, _, lhs_args, _) = &lhs.kind;
            if arg_lhs_path.ident.name == sym::len;
            if let Some(arg_lhs_struct) = lhs_args.get(0);

            // The two vectors referenced (x in x.get(...) and in x.len())
            if SpanlessEq::new(cx).eq_expr(struct_calling_on, arg_lhs_struct);

            // RHS of subtraction is 1
            if let ExprKind::Lit(rhs_lit) = &rhs.kind;
            if let LitKind::Int(1, ..) = rhs_lit.node;

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
