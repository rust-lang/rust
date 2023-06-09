use clippy_utils::{diagnostics::span_lint_and_sugg, ty::is_type_lang_item};
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, LangItem, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Detects cases of owned empty strings being passed as an argument to a function expecting `&str`
    ///
    /// ### Why is this bad?
    ///
    /// This results in longer and less readable code
    ///
    /// ### Example
    /// ```rust
    /// vec!["1", "2", "3"].join(&String::new());
    /// ```
    /// Use instead:
    /// ```rust
    /// vec!["1", "2", "3"].join("");
    /// ```
    #[clippy::version = "1.62.0"]
    pub UNNECESSARY_OWNED_EMPTY_STRINGS,
    style,
    "detects cases of references to owned empty strings being passed as an argument to a function expecting `&str`"
}
declare_lint_pass!(UnnecessaryOwnedEmptyStrings => [UNNECESSARY_OWNED_EMPTY_STRINGS]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryOwnedEmptyStrings {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if_chain! {
            if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner_expr) = expr.kind;
            if let ExprKind::Call(fun, args) = inner_expr.kind;
            if let ExprKind::Path(ref qpath) = fun.kind;
            if let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id();
            if let ty::Ref(_, inner_str, _) = cx.typeck_results().expr_ty_adjusted(expr).kind();
            if inner_str.is_str();
            then {
                if match_def_path(cx, fun_def_id, &paths::STRING_NEW) {
                     span_lint_and_sugg(
                            cx,
                            UNNECESSARY_OWNED_EMPTY_STRINGS,
                            expr.span,
                            "usage of `&String::new()` for a function expecting a `&str` argument",
                            "try",
                            "\"\"".to_owned(),
                            Applicability::MachineApplicable,
                        );
                } else {
                    if_chain! {
                        if cx.tcx.is_diagnostic_item(sym::from_fn, fun_def_id);
                        if let [.., last_arg] = args;
                        if let ExprKind::Lit(spanned) = &last_arg.kind;
                        if let LitKind::Str(symbol, _) = spanned.node;
                        if symbol.is_empty();
                        let inner_expr_type = cx.typeck_results().expr_ty(inner_expr);
                        if is_type_lang_item(cx, inner_expr_type, LangItem::String);
                        then {
                            span_lint_and_sugg(
                                cx,
                                UNNECESSARY_OWNED_EMPTY_STRINGS,
                                expr.span,
                                "usage of `&String::from(\"\")` for a function expecting a `&str` argument",
                                "try",
                                "\"\"".to_owned(),
                                Applicability::MachineApplicable,
                            );
                        }
                    }
                }
            }
        }
    }
}
