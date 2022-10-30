use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{match_def_path, paths, peel_hir_expr_refs};
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Detects cases where the result of a `format!` call is
    /// appended to an existing `String`.
    ///
    /// ### Why is this bad?
    /// Introduces an extra, avoidable heap allocation.
    ///
    /// ### Known problems
    /// `format!` returns a `String` but `write!` returns a `Result`.
    /// Thus you are forced to ignore the `Err` variant to achieve the same API.
    ///
    /// While using `write!` in the suggested way should never fail, this isn't necessarily clear to the programmer.
    ///
    /// ### Example
    /// ```rust
    /// let mut s = String::new();
    /// s += &format!("0x{:X}", 1024);
    /// s.push_str(&format!("0x{:X}", 1024));
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::fmt::Write as _; // import without risk of name clashing
    ///
    /// let mut s = String::new();
    /// let _ = write!(s, "0x{:X}", 1024);
    /// ```
    #[clippy::version = "1.62.0"]
    pub FORMAT_PUSH_STRING,
    restriction,
    "`format!(..)` appended to existing `String`"
}
declare_lint_pass!(FormatPushString => [FORMAT_PUSH_STRING]);

fn is_string(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(e).peel_refs(), sym::String)
}
fn is_format(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    if let Some(macro_def_id) = e.span.ctxt().outer_expn_data().macro_def_id {
        cx.tcx.get_diagnostic_name(macro_def_id) == Some(sym::format_macro)
    } else {
        false
    }
}

impl<'tcx> LateLintPass<'tcx> for FormatPushString {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let arg = match expr.kind {
            ExprKind::MethodCall(_, _, [arg], _) => {
                if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) &&
                match_def_path(cx, fn_def_id, &paths::PUSH_STR) {
                    arg
                } else {
                    return;
                }
            }
            ExprKind::AssignOp(op, left, arg)
            if op.node == BinOpKind::Add && is_string(cx, left) => {
                arg
            },
            _ => return,
        };
        let (arg, _) = peel_hir_expr_refs(arg);
        if is_format(cx, arg) {
            span_lint_and_help(
                cx,
                FORMAT_PUSH_STRING,
                expr.span,
                "`format!(..)` appended to existing `String`",
                None,
                "consider using `write!` to avoid the extra allocation",
            );
        }
    }
}
