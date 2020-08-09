use crate::utils::{match_def_path, paths, snippet_with_applicability, span_lint_and_sugg};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Warns when using push_str with a single-character string literal,
    /// and push with a char would work fine.
    ///
    /// **Why is this bad?** This is in all probability not the intended outcome. At
    /// the least it hurts readability of the code.
    ///
    /// **Known problems:** None
    ///
    /// **Example:**
    /// ```
    /// let mut string = String::new();
    /// string.push_str("R");
    /// ```
    /// Could be written as
    /// ```
    /// let mut string = String::new();
    /// string.push('R');
    /// ```
    pub SINGLE_CHAR_PUSH_STR,
    style,
    "`push_str()` used with a single-character string literal as parameter"
}

declare_lint_pass!(SingleCharPushStrPass => [SINGLE_CHAR_PUSH_STR]);

impl<'tcx> LateLintPass<'tcx> for SingleCharPushStrPass {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(_, _, ref args, _) = expr.kind;
            if let [base_string, extension_string] = args;
            if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
            if match_def_path(cx, fn_def_id, &paths::PUSH_STR);
            if let ExprKind::Lit(ref lit) = extension_string.kind;
            if let LitKind::Str(symbol,_) = lit.node;
            let extension_string_val = symbol.as_str().to_string();
            if extension_string_val.len() == 1;
            then {
                let mut applicability = Applicability::MachineApplicable;
                let base_string_snippet = snippet_with_applicability(cx, base_string.span, "_", &mut applicability);
                let sugg = format!("{}.push({:?})", base_string_snippet, extension_string_val.chars().next().unwrap());
                span_lint_and_sugg(
                    cx,
                    SINGLE_CHAR_PUSH_STR,
                    expr.span,
                    "calling `push_str()` using a single-character string literal",
                    "consider using `push` with a character literal",
                    sugg,
                    applicability
                );
            }
        }
    }
}
