use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher;
use clippy_utils::ty::is_type_lang_item;
use rustc_hir::{AssignOpKind, Expr, ExprKind, LangItem, MatchSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
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
    /// ```no_run
    /// let mut s = String::new();
    /// s += &format!("0x{:X}", 1024);
    /// s.push_str(&format!("0x{:X}", 1024));
    /// ```
    /// Use instead:
    /// ```no_run
    /// use std::fmt::Write as _; // import without risk of name clashing
    ///
    /// let mut s = String::new();
    /// let _ = write!(s, "0x{:X}", 1024);
    /// ```
    #[clippy::version = "1.62.0"]
    pub FORMAT_PUSH_STRING,
    pedantic,
    "`format!(..)` appended to existing `String`"
}
declare_lint_pass!(FormatPushString => [FORMAT_PUSH_STRING]);

fn is_string(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    is_type_lang_item(cx, cx.typeck_results().expr_ty(e).peel_refs(), LangItem::String)
}
fn is_format(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    let e = e.peel_blocks().peel_borrows();

    if e.span.from_expansion()
        && let Some(macro_def_id) = e.span.ctxt().outer_expn_data().macro_def_id
    {
        cx.tcx.get_diagnostic_name(macro_def_id) == Some(sym::format_macro)
    } else if let Some(higher::If { then, r#else, .. }) = higher::If::hir(e) {
        is_format(cx, then) || r#else.is_some_and(|e| is_format(cx, e))
    } else {
        match higher::IfLetOrMatch::parse(cx, e) {
            Some(higher::IfLetOrMatch::Match(_, arms, MatchSource::Normal)) => {
                arms.iter().any(|arm| is_format(cx, arm.body))
            },
            Some(higher::IfLetOrMatch::IfLet(_, _, then, r#else, _)) => {
                is_format(cx, then) || r#else.is_some_and(|e| is_format(cx, e))
            },
            _ => false,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for FormatPushString {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        let arg = match expr.kind {
            ExprKind::MethodCall(_, _, [arg], _) => {
                if let Some(fn_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                    && cx.tcx.is_diagnostic_item(sym::string_push_str, fn_def_id)
                {
                    arg
                } else {
                    return;
                }
            },
            ExprKind::AssignOp(op, left, arg) if op.node == AssignOpKind::AddAssign && is_string(cx, left) => arg,
            _ => return,
        };
        if is_format(cx, arg) {
            #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
            span_lint_and_then(
                cx,
                FORMAT_PUSH_STRING,
                expr.span,
                "`format!(..)` appended to existing `String`",
                |diag| {
                    diag.help("consider using `write!` to avoid the extra allocation");
                },
            );
        }
    }
}
