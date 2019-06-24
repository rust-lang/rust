use crate::utils::{differing_macro_contexts, in_macro_or_desugar, snippet_opt, span_note_and_lint};
use if_chain::if_chain;
use rustc::lint::{in_external_macro, EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::ast;
use syntax::ptr::P;

declare_clippy_lint! {
    /// **What it does:** Checks for use of the non-existent `=*`, `=!` and `=-`
    /// operators.
    ///
    /// **Why is this bad?** This is either a typo of `*=`, `!=` or `-=` or
    /// confusing.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// a =- 42; // confusing, should it be `a -= 42` or `a = -42`?
    /// ```
    pub SUSPICIOUS_ASSIGNMENT_FORMATTING,
    style,
    "suspicious formatting of `*=`, `-=` or `!=`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for formatting of `else`. It lints if the `else`
    /// is followed immediately by a newline or the `else` seems to be missing.
    ///
    /// **Why is this bad?** This is probably some refactoring remnant, even if the
    /// code is correct, it might look confusing.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// if foo {
    /// } { // looks like an `else` is missing here
    /// }
    ///
    /// if foo {
    /// } if bar { // looks like an `else` is missing here
    /// }
    ///
    /// if foo {
    /// } else
    ///
    /// { // this is the `else` block of the previous `if`, but should it be?
    /// }
    ///
    /// if foo {
    /// } else
    ///
    /// if bar { // this is the `else` block of the previous `if`, but should it be?
    /// }
    /// ```
    pub SUSPICIOUS_ELSE_FORMATTING,
    style,
    "suspicious formatting of `else`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for possible missing comma in an array. It lints if
    /// an array element is a binary operator expression and it lies on two lines.
    ///
    /// **Why is this bad?** This could lead to unexpected results.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// let a = &[
    ///     -1, -2, -3 // <= no comma here
    ///     -4, -5, -6
    /// ];
    /// ```
    pub POSSIBLE_MISSING_COMMA,
    correctness,
    "possible missing comma in array"
}

declare_lint_pass!(Formatting => [
    SUSPICIOUS_ASSIGNMENT_FORMATTING,
    SUSPICIOUS_ELSE_FORMATTING,
    POSSIBLE_MISSING_COMMA
]);

impl EarlyLintPass for Formatting {
    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &ast::Block) {
        for w in block.stmts.windows(2) {
            match (&w[0].node, &w[1].node) {
                (&ast::StmtKind::Expr(ref first), &ast::StmtKind::Expr(ref second))
                | (&ast::StmtKind::Expr(ref first), &ast::StmtKind::Semi(ref second)) => {
                    check_missing_else(cx, first, second);
                },
                _ => (),
            }
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        check_assign(cx, expr);
        check_else(cx, expr);
        check_array(cx, expr);
    }
}

/// Implementation of the `SUSPICIOUS_ASSIGNMENT_FORMATTING` lint.
fn check_assign(cx: &EarlyContext<'_>, expr: &ast::Expr) {
    if let ast::ExprKind::Assign(ref lhs, ref rhs) = expr.node {
        if !differing_macro_contexts(lhs.span, rhs.span) && !in_macro_or_desugar(lhs.span) {
            let eq_span = lhs.span.between(rhs.span);
            if let ast::ExprKind::Unary(op, ref sub_rhs) = rhs.node {
                if let Some(eq_snippet) = snippet_opt(cx, eq_span) {
                    let op = ast::UnOp::to_string(op);
                    let eqop_span = lhs.span.between(sub_rhs.span);
                    if eq_snippet.ends_with('=') {
                        span_note_and_lint(
                            cx,
                            SUSPICIOUS_ASSIGNMENT_FORMATTING,
                            eqop_span,
                            &format!(
                                "this looks like you are trying to use `.. {op}= ..`, but you \
                                 really are doing `.. = ({op} ..)`",
                                op = op
                            ),
                            eqop_span,
                            &format!("to remove this lint, use either `{op}=` or `= {op}`", op = op),
                        );
                    }
                }
            }
        }
    }
}

/// Implementation of the `SUSPICIOUS_ELSE_FORMATTING` lint for weird `else`.
fn check_else(cx: &EarlyContext<'_>, expr: &ast::Expr) {
    if_chain! {
        if let Some((then, &Some(ref else_))) = unsugar_if(expr);
        if is_block(else_) || unsugar_if(else_).is_some();
        if !differing_macro_contexts(then.span, else_.span);
        if !in_macro_or_desugar(then.span) && !in_external_macro(cx.sess, expr.span);

        // workaround for rust-lang/rust#43081
        if expr.span.lo().0 != 0 && expr.span.hi().0 != 0;

        // this will be a span from the closing ‘}’ of the “then” block (excluding) to
        // the “if” of the “else if” block (excluding)
        let else_span = then.span.between(else_.span);

        // the snippet should look like " else \n    " with maybe comments anywhere
        // it’s bad when there is a ‘\n’ after the “else”
        if let Some(else_snippet) = snippet_opt(cx, else_span);
        if let Some(else_pos) = else_snippet.find("else");
        if else_snippet[else_pos..].contains('\n');
        let else_desc = if unsugar_if(else_).is_some() { "if" } else { "{..}" };

        then {
            span_note_and_lint(
                cx,
                SUSPICIOUS_ELSE_FORMATTING,
                else_span,
                &format!("this is an `else {}` but the formatting might hide it", else_desc),
                else_span,
                &format!(
                    "to remove this lint, remove the `else` or remove the new line between \
                     `else` and `{}`",
                    else_desc,
                ),
            );
        }
    }
}

fn has_unary_equivalent(bin_op: ast::BinOpKind) -> bool {
    // &, *, -
    bin_op == ast::BinOpKind::And || bin_op == ast::BinOpKind::Mul || bin_op == ast::BinOpKind::Sub
}

/// Implementation of the `POSSIBLE_MISSING_COMMA` lint for array
fn check_array(cx: &EarlyContext<'_>, expr: &ast::Expr) {
    if let ast::ExprKind::Array(ref array) = expr.node {
        for element in array {
            if let ast::ExprKind::Binary(ref op, ref lhs, _) = element.node {
                if has_unary_equivalent(op.node) && !differing_macro_contexts(lhs.span, op.span) {
                    let space_span = lhs.span.between(op.span);
                    if let Some(space_snippet) = snippet_opt(cx, space_span) {
                        let lint_span = lhs.span.with_lo(lhs.span.hi());
                        if space_snippet.contains('\n') {
                            span_note_and_lint(
                                cx,
                                POSSIBLE_MISSING_COMMA,
                                lint_span,
                                "possibly missing a comma here",
                                lint_span,
                                "to remove this lint, add a comma or write the expr in a single line",
                            );
                        }
                    }
                }
            }
        }
    }
}

fn check_missing_else(cx: &EarlyContext<'_>, first: &ast::Expr, second: &ast::Expr) {
    if !differing_macro_contexts(first.span, second.span)
        && !in_macro_or_desugar(first.span)
        && unsugar_if(first).is_some()
        && (is_block(second) || unsugar_if(second).is_some())
    {
        // where the else would be
        let else_span = first.span.between(second.span);

        if let Some(else_snippet) = snippet_opt(cx, else_span) {
            if !else_snippet.contains('\n') {
                let (looks_like, next_thing) = if unsugar_if(second).is_some() {
                    ("an `else if`", "the second `if`")
                } else {
                    ("an `else {..}`", "the next block")
                };

                span_note_and_lint(
                    cx,
                    SUSPICIOUS_ELSE_FORMATTING,
                    else_span,
                    &format!("this looks like {} but the `else` is missing", looks_like),
                    else_span,
                    &format!(
                        "to remove this lint, add the missing `else` or add a new line before {}",
                        next_thing,
                    ),
                );
            }
        }
    }
}

fn is_block(expr: &ast::Expr) -> bool {
    if let ast::ExprKind::Block(..) = expr.node {
        true
    } else {
        false
    }
}

/// Match `if` or `if let` expressions and return the `then` and `else` block.
fn unsugar_if(expr: &ast::Expr) -> Option<(&P<ast::Block>, &Option<P<ast::Expr>>)> {
    match expr.node {
        ast::ExprKind::If(_, ref then, ref else_) => Some((then, else_)),
        _ => None,
    }
}
