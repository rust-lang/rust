use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::HasSession;
use clippy_utils::sugg::Sugg;
use clippy_utils::{higher, is_else_clause, is_in_const_context, span_contains_comment};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Instead of using an if statement to convert a bool to an int,
    /// this lint suggests using a `from()` function or an `as` coercion.
    ///
    /// ### Why is this bad?
    /// Coercion or `from()` is another way to convert bool to a number.
    /// Both methods are guaranteed to return 1 for true, and 0 for false.
    ///
    /// See https://doc.rust-lang.org/std/primitive.bool.html#impl-From%3Cbool%3E
    ///
    /// ### Example
    /// ```no_run
    /// # let condition = false;
    /// if condition {
    ///     1_i64
    /// } else {
    ///     0
    /// };
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let condition = false;
    /// i64::from(condition);
    /// ```
    /// or
    /// ```no_run
    /// # let condition = false;
    /// condition as i64;
    /// ```
    #[clippy::version = "1.65.0"]
    pub BOOL_TO_INT_WITH_IF,
    pedantic,
    "using if to convert bool to int"
}
declare_lint_pass!(BoolToIntWithIf => [BOOL_TO_INT_WITH_IF]);

impl<'tcx> LateLintPass<'tcx> for BoolToIntWithIf {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if !expr.span.from_expansion()
            && let Some(higher::If {
                cond,
                then,
                r#else: Some(r#else),
            }) = higher::If::hir(expr)
            && let Some(then_lit) = as_int_bool_lit(then)
            && let Some(else_lit) = as_int_bool_lit(r#else)
            && then_lit != else_lit
            && !is_in_const_context(cx)
        {
            let ty = cx.typeck_results().expr_ty(then);
            let mut applicability = if span_contains_comment(cx.sess().source_map(), expr.span) {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };
            let snippet = {
                let mut sugg = Sugg::hir_with_context(cx, cond, expr.span.ctxt(), "..", &mut applicability);
                if !then_lit {
                    sugg = !sugg;
                }
                sugg
            };
            let suggestion = {
                let mut s = Sugg::NonParen(format!("{ty}::from({snippet})").into());
                // when used in else clause if statement should be wrapped in curly braces
                if is_else_clause(cx.tcx, expr) {
                    s = s.blockify();
                }
                s
            };

            let into_snippet = snippet.clone().maybe_paren();
            let as_snippet = snippet.as_ty(ty);

            span_lint_and_then(
                cx,
                BOOL_TO_INT_WITH_IF,
                expr.span,
                "boolean to int conversion using if",
                |diag| {
                    diag.span_suggestion(expr.span, "replace with from", suggestion, applicability);
                    diag.note(format!(
                        "`{as_snippet}` or `{into_snippet}.into()` can also be valid options"
                    ));
                },
            );
        }
    }
}

fn as_int_bool_lit(expr: &Expr<'_>) -> Option<bool> {
    if let ExprKind::Block(b, _) = expr.kind
        && b.stmts.is_empty()
        && let Some(e) = b.expr
        && !e.span.from_expansion()
        && let ExprKind::Lit(lit) = e.kind
        && let LitKind::Int(x, _) = lit.node
    {
        match x.get() {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    } else {
        None
    }
}
