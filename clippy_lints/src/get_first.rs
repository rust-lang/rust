use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_slice_of_primitives, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Spanned;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for using `x.get(0)` instead of
    /// `x.first()`.
    ///
    /// ### Why is this bad?
    /// Using `x.first()` is easier to read and has the same
    /// result.
    ///
    /// ### Example
    /// ```rust
    /// // Bad
    /// let x = vec![2, 3, 5];
    /// let first_element = x.get(0);
    /// ```
    /// Use instead:
    /// ```rust
    /// // Good
    /// let x = vec![2, 3, 5];
    /// let first_element = x.first();
    /// ```
    #[clippy::version = "1.63.0"]
    pub GET_FIRST,
    style,
    "Using `x.get(0)` when `x.first()` is simpler"
}
declare_lint_pass!(GetFirst => [GET_FIRST]);

impl<'tcx> LateLintPass<'tcx> for GetFirst {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            if let hir::ExprKind::MethodCall(_, [struct_calling_on, method_arg], _) = &expr.kind;
            if let Some(expr_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if match_def_path(cx, expr_def_id, &paths::SLICE_GET);

            if let Some(_) = is_slice_of_primitives(cx, struct_calling_on);
            if let hir::ExprKind::Lit(Spanned { node: LitKind::Int(0, _), .. }) = method_arg.kind;

            then {
                let mut applicability = Applicability::MachineApplicable;
                let slice_name = snippet_with_applicability(
                    cx,
                    struct_calling_on.span, "..",
                    &mut applicability,
                );
                span_lint_and_sugg(
                    cx,
                    GET_FIRST,
                    expr.span,
                    &format!("accessing first element with `{0}.get(0)`", slice_name),
                    "try",
                    format!("{}.first()", slice_name),
                    applicability,
                );
            }
        }
    }
}
