use clippy_utils::higher::If;
use rustc_ast::LitKind;
use rustc_hir::{Block, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg::Sugg;
use clippy_utils::{in_constant, is_else_clause, is_integer_literal};
use rustc_errors::Applicability;

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
    /// ```rust
    /// # let condition = false;
    /// if condition {
    ///     1_i64
    /// } else {
    ///     0
    /// };
    /// ```
    /// Use instead:
    /// ```rust
    /// # let condition = false;
    /// i64::from(condition);
    /// ```
    /// or
    /// ```rust
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
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx rustc_hir::Expr<'tcx>) {
        if !expr.span.from_expansion() && !in_constant(cx, expr.hir_id) {
            check_if_else(cx, expr);
        }
    }
}

fn check_if_else<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx rustc_hir::Expr<'tcx>) {
    if let Some(If { cond, then, r#else: Some(r#else) }) = If::hir(expr)
        && let Some(then_lit) = int_literal(then)
        && let Some(else_lit) = int_literal(r#else)
    {
        let inverted = if is_integer_literal(then_lit, 1) && is_integer_literal(else_lit, 0) {
            false
        } else if is_integer_literal(then_lit, 0) && is_integer_literal(else_lit, 1) {
            true
        } else {
            // Expression isn't boolean, exit
            return;
        };
        let mut applicability = Applicability::MachineApplicable;
        let snippet = {
            let mut sugg = Sugg::hir_with_applicability(cx, cond, "..", &mut applicability);
            if inverted {
                sugg = !sugg;
            }
            sugg
        };

        let ty = cx.typeck_results().expr_ty(then_lit); // then and else must be of same type

        let suggestion = {
            let wrap_in_curly = is_else_clause(cx.tcx, expr);
            let mut s = Sugg::NonParen(format!("{ty}::from({snippet})").into());
            if wrap_in_curly {
                s = s.blockify();
            }
            s
        }; // when used in else clause if statement should be wrapped in curly braces

        let into_snippet = snippet.clone().maybe_par();
        let as_snippet = snippet.as_ty(ty);

        span_lint_and_then(cx,
            BOOL_TO_INT_WITH_IF,
            expr.span,
            "boolean to int conversion using if",
            |diag| {
            diag.span_suggestion(
                expr.span,
                "replace with from",
                suggestion,
                applicability,
            );
            diag.note(format!("`{as_snippet}` or `{into_snippet}.into()` can also be valid options"));
        });
    };
}

// If block contains only a int literal expression, return literal expression
fn int_literal<'tcx>(expr: &'tcx rustc_hir::Expr<'tcx>) -> Option<&'tcx rustc_hir::Expr<'tcx>> {
    if let ExprKind::Block(block, _) = expr.kind
        && let Block {
            stmts: [],       // Shouldn't lint if statements with side effects
            expr: Some(expr),
            ..
        } = block
        && let ExprKind::Lit(lit) = &expr.kind
        && let LitKind::Int(_, _) = lit.node
    {
        Some(expr)
    } else {
        None
    }
}
