use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// It checks for `str::bytes().count()` and suggests replacing it with
    /// `str::len()`.
    ///
    /// ### Why is this bad?
    /// `str::bytes().count()` is longer and may not be as performant as using
    /// `str::len()`.
    ///
    /// ### Example
    /// ```rust
    /// "hello".bytes().count();
    /// String::from("hello").bytes().count();
    /// ```
    /// Use instead:
    /// ```rust
    /// "hello".len();
    /// String::from("hello").len();
    /// ```
    #[clippy::version = "1.62.0"]
    pub BYTES_COUNT_TO_LEN,
    complexity,
    "Using `bytes().count()` when `len()` performs the same functionality"
}

declare_lint_pass!(BytesCountToLen => [BYTES_COUNT_TO_LEN]);

impl<'tcx> LateLintPass<'tcx> for BytesCountToLen {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if_chain! {
            if let hir::ExprKind::MethodCall(_, expr_args, _) = &expr.kind;
            if let Some(expr_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if match_def_path(cx, expr_def_id, &paths::ITER_COUNT);

            if let [bytes_expr] = &**expr_args;
            if let hir::ExprKind::MethodCall(_, bytes_args, _) = &bytes_expr.kind;
            if let Some(bytes_def_id) = cx.typeck_results().type_dependent_def_id(bytes_expr.hir_id);
            if match_def_path(cx, bytes_def_id, &paths::STR_BYTES);

            if let [str_expr] = &**bytes_args;
            let ty = cx.typeck_results().expr_ty(str_expr).peel_refs();

            if is_type_diagnostic_item(cx, ty, sym::String) || ty.kind() == &ty::Str;
            then {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    BYTES_COUNT_TO_LEN,
                    expr.span,
                    "using long and hard to read `.bytes().count()`",
                    "consider calling `.len()` instead",
                    format!("{}.len()", snippet_with_applicability(cx, str_expr.span, "..", &mut applicability)),
                    applicability
                );
            }
        };
    }
}
