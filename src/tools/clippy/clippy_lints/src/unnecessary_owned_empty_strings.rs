use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_lang_item;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BorrowKind, Expr, ExprKind, LangItem, Mutability};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::declare_lint_pass;
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
    /// ```no_run
    /// vec!["1", "2", "3"].join(&String::new());
    /// ```
    /// Use instead:
    /// ```no_run
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
        if let ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, inner_expr) = expr.kind
            && let ExprKind::Call(fun, args) = inner_expr.kind
            && let ExprKind::Path(ref qpath) = fun.kind
            && let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id()
            && let ty::Ref(_, inner_str, _) = cx.typeck_results().expr_ty_adjusted(expr).kind()
            && inner_str.is_str()
        {
            let fun_name = cx.tcx.get_diagnostic_name(fun_def_id);
            if fun_name == Some(sym::string_new) {
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_OWNED_EMPTY_STRINGS,
                    expr.span,
                    "usage of `&String::new()` for a function expecting a `&str` argument",
                    "try",
                    "\"\"".to_owned(),
                    Applicability::MachineApplicable,
                );
            } else if fun_name == Some(sym::from_fn)
                && let [arg] = args
                && let ExprKind::Lit(spanned) = &arg.kind
                && let LitKind::Str(symbol, _) = spanned.node
                && symbol.is_empty()
                && let inner_expr_type = cx.typeck_results().expr_ty(inner_expr)
                && is_type_lang_item(cx, inner_expr_type, LangItem::String)
            {
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
