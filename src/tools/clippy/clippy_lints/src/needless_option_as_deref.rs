use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for no-op uses of Option::{as_deref,as_deref_mut},
    /// for example, `Option<&T>::as_deref()` returns the same type.
    ///
    /// ### Why is this bad?
    /// Redundant code and improving readability.
    ///
    /// ### Example
    /// ```rust
    /// let a = Some(&1);
    /// let b = a.as_deref(); // goes from Option<&i32> to Option<&i32>
    /// ```
    /// Could be written as:
    /// ```rust
    /// let a = Some(&1);
    /// let b = a;
    /// ```
    #[clippy::version = "1.57.0"]
    pub NEEDLESS_OPTION_AS_DEREF,
    complexity,
    "no-op use of `deref` or `deref_mut` method to `Option`."
}

declare_lint_pass!(OptionNeedlessDeref=> [
    NEEDLESS_OPTION_AS_DEREF,
]);

impl<'tcx> LateLintPass<'tcx> for OptionNeedlessDeref {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.from_expansion() {
            return;
        }
        let typeck = cx.typeck_results();
        let outer_ty = typeck.expr_ty(expr);

        if_chain! {
            if is_type_diagnostic_item(cx,outer_ty,sym::Option);
            if let ExprKind::MethodCall(path, [sub_expr], _) = expr.kind;
            let symbol = path.ident.as_str();
            if symbol == "as_deref" || symbol == "as_deref_mut";
            if outer_ty == typeck.expr_ty(sub_expr);
            then{
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_OPTION_AS_DEREF,
                    expr.span,
                    "derefed type is same as origin",
                    "try this",
                    snippet_opt(cx,sub_expr.span).unwrap(),
                    Applicability::MachineApplicable
                );
            }
        }
    }
}
