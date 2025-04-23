use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::match_type;
use clippy_utils::{is_lint_allowed, method_calls, paths};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_session::declare_lint_pass;
use rustc_span::symbol::Symbol;

declare_tool_lint! {
    /// ### What it does
    /// Checks for calls to `cx.outer().expn_data()` and suggests to use
    /// the `cx.outer_expn_data()`
    ///
    /// ### Why is this bad?
    /// `cx.outer_expn_data()` is faster and more concise.
    ///
    /// ### Example
    /// ```rust,ignore
    /// expr.span.ctxt().outer().expn_data()
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// expr.span.ctxt().outer_expn_data()
    /// ```
    pub clippy::OUTER_EXPN_EXPN_DATA,
    Warn,
    "using `cx.outer_expn().expn_data()` instead of `cx.outer_expn_data()`",
    report_in_external_macro: true
}

declare_lint_pass!(OuterExpnDataPass => [OUTER_EXPN_EXPN_DATA]);

impl<'tcx> LateLintPass<'tcx> for OuterExpnDataPass {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if is_lint_allowed(cx, OUTER_EXPN_EXPN_DATA, expr.hir_id) {
            return;
        }

        let (method_names, arg_lists, spans) = method_calls(expr, 2);
        let method_names: Vec<&str> = method_names.iter().map(Symbol::as_str).collect();
        if let ["expn_data", "outer_expn"] = method_names.as_slice()
            && let (self_arg, args) = arg_lists[1]
            && args.is_empty()
            && let self_ty = cx.typeck_results().expr_ty(self_arg).peel_refs()
            && match_type(cx, self_ty, &paths::SYNTAX_CONTEXT)
        {
            span_lint_and_sugg(
                cx,
                OUTER_EXPN_EXPN_DATA,
                spans[1].with_hi(expr.span.hi()),
                "usage of `outer_expn().expn_data()`",
                "try",
                "outer_expn_data()".to_string(),
                Applicability::MachineApplicable,
            );
        }
    }
}
