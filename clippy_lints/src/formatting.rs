use crate::utils::{differing_macro_contexts, snippet_opt, span_lint_and_help, span_lint_and_note};
use if_chain::if_chain;
use rustc_ast::ast::{BinOpKind, Block, Expr, ExprKind, StmtKind, UnOp};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

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
    /// **What it does:** Checks the formatting of a unary operator on the right hand side
    /// of a binary operator. It lints if there is no space between the binary and unary operators,
    /// but there is a space between the unary and its operand.
    ///
    /// **Why is this bad?** This is either a typo in the binary operator or confusing.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// if foo <- 30 { // this should be `foo < -30` but looks like a different operator
    /// }
    ///
    /// if foo &&! bar { // this should be `foo && !bar` but looks like a different operator
    /// }
    /// ```
    pub SUSPICIOUS_UNARY_OP_FORMATTING,
    style,
    "suspicious formatting of unary `-` or `!` on the RHS of a BinOp"
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
    SUSPICIOUS_UNARY_OP_FORMATTING,
    SUSPICIOUS_ELSE_FORMATTING,
    POSSIBLE_MISSING_COMMA
]);

impl EarlyLintPass for Formatting {
    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &Block) {
        for w in block.stmts.windows(2) {
            if let (StmtKind::Expr(first), StmtKind::Expr(second) | StmtKind::Semi(second)) = (&w[0].kind, &w[1].kind) {
                check_missing_else(cx, first, second);
            }
        }
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &Expr) {
        check_assign(cx, expr);
        check_unop(cx, expr);
        check_else(cx, expr);
        check_array(cx, expr);
    }
}

/// Implementation of the `SUSPICIOUS_ASSIGNMENT_FORMATTING` lint.
fn check_assign(cx: &EarlyContext<'_>, expr: &Expr) {
    if let ExprKind::Assign(ref lhs, ref rhs, _) = expr.kind {
        if !differing_macro_contexts(lhs.span, rhs.span) && !lhs.span.from_expansion() {
            let eq_span = lhs.span.between(rhs.span);
            if let ExprKind::Unary(op, ref sub_rhs) = rhs.kind {
                if let Some(eq_snippet) = snippet_opt(cx, eq_span) {
                    let op = UnOp::to_string(op);
                    let eqop_span = lhs.span.between(sub_rhs.span);
                    if eq_snippet.ends_with('=') {
                        span_lint_and_note(
                            cx,
                            SUSPICIOUS_ASSIGNMENT_FORMATTING,
                            eqop_span,
                            &format!(
                                "this looks like you are trying to use `.. {op}= ..`, but you \
                                 really are doing `.. = ({op} ..)`",
                                op = op
                            ),
                            None,
                            &format!("to remove this lint, use either `{op}=` or `= {op}`", op = op),
                        );
                    }
                }
            }
        }
    }
}

/// Implementation of the `SUSPICIOUS_UNARY_OP_FORMATTING` lint.
fn check_unop(cx: &EarlyContext<'_>, expr: &Expr) {
    if_chain! {
        if let ExprKind::Binary(ref binop, ref lhs, ref rhs) = expr.kind;
        if !differing_macro_contexts(lhs.span, rhs.span) && !lhs.span.from_expansion();
        // span between BinOp LHS and RHS
        let binop_span = lhs.span.between(rhs.span);
        // if RHS is a UnOp
        if let ExprKind::Unary(op, ref un_rhs) = rhs.kind;
        // from UnOp operator to UnOp operand
        let unop_operand_span = rhs.span.until(un_rhs.span);
        if let Some(binop_snippet) = snippet_opt(cx, binop_span);
        if let Some(unop_operand_snippet) = snippet_opt(cx, unop_operand_span);
        let binop_str = BinOpKind::to_string(&binop.node);
        // no space after BinOp operator and space after UnOp operator
        if binop_snippet.ends_with(binop_str) && unop_operand_snippet.ends_with(' ');
        then {
            let unop_str = UnOp::to_string(op);
            let eqop_span = lhs.span.between(un_rhs.span);
            span_lint_and_help(
                cx,
                SUSPICIOUS_UNARY_OP_FORMATTING,
                eqop_span,
                &format!(
                    "by not having a space between `{binop}` and `{unop}` it looks like \
                     `{binop}{unop}` is a single operator",
                    binop = binop_str,
                    unop = unop_str
                ),
                None,
                &format!(
                    "put a space between `{binop}` and `{unop}` and remove the space after `{unop}`",
                    binop = binop_str,
                    unop = unop_str
                ),
            );
        }
    }
}

/// Implementation of the `SUSPICIOUS_ELSE_FORMATTING` lint for weird `else`.
fn check_else(cx: &EarlyContext<'_>, expr: &Expr) {
    if_chain! {
        if let ExprKind::If(_, then, Some(else_)) = &expr.kind;
        if is_block(else_) || is_if(else_);
        if !differing_macro_contexts(then.span, else_.span);
        if !then.span.from_expansion() && !in_external_macro(cx.sess, expr.span);

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
        let else_desc = if is_if(else_) { "if" } else { "{..}" };

        then {
            span_lint_and_note(
                cx,
                SUSPICIOUS_ELSE_FORMATTING,
                else_span,
                &format!("this is an `else {}` but the formatting might hide it", else_desc),
                None,
                &format!(
                    "to remove this lint, remove the `else` or remove the new line between \
                     `else` and `{}`",
                    else_desc,
                ),
            );
        }
    }
}

#[must_use]
fn has_unary_equivalent(bin_op: BinOpKind) -> bool {
    // &, *, -
    bin_op == BinOpKind::And || bin_op == BinOpKind::Mul || bin_op == BinOpKind::Sub
}

fn indentation(cx: &EarlyContext<'_>, span: Span) -> usize {
    cx.sess.source_map().lookup_char_pos(span.lo()).col.0
}

/// Implementation of the `POSSIBLE_MISSING_COMMA` lint for array
fn check_array(cx: &EarlyContext<'_>, expr: &Expr) {
    if let ExprKind::Array(ref array) = expr.kind {
        for element in array {
            if_chain! {
                if let ExprKind::Binary(ref op, ref lhs, _) = element.kind;
                if has_unary_equivalent(op.node) && !differing_macro_contexts(lhs.span, op.span);
                let space_span = lhs.span.between(op.span);
                if let Some(space_snippet) = snippet_opt(cx, space_span);
                let lint_span = lhs.span.with_lo(lhs.span.hi());
                if space_snippet.contains('\n');
                if indentation(cx, op.span) <= indentation(cx, lhs.span);
                then {
                    span_lint_and_note(
                        cx,
                        POSSIBLE_MISSING_COMMA,
                        lint_span,
                        "possibly missing a comma here",
                        None,
                        "to remove this lint, add a comma or write the expr in a single line",
                    );
                }
            }
        }
    }
}

fn check_missing_else(cx: &EarlyContext<'_>, first: &Expr, second: &Expr) {
    if !differing_macro_contexts(first.span, second.span)
        && !first.span.from_expansion()
        && is_if(first)
        && (is_block(second) || is_if(second))
    {
        // where the else would be
        let else_span = first.span.between(second.span);

        if let Some(else_snippet) = snippet_opt(cx, else_span) {
            if !else_snippet.contains('\n') {
                let (looks_like, next_thing) = if is_if(second) {
                    ("an `else if`", "the second `if`")
                } else {
                    ("an `else {..}`", "the next block")
                };

                span_lint_and_note(
                    cx,
                    SUSPICIOUS_ELSE_FORMATTING,
                    else_span,
                    &format!("this looks like {} but the `else` is missing", looks_like),
                    None,
                    &format!(
                        "to remove this lint, add the missing `else` or add a new line before {}",
                        next_thing,
                    ),
                );
            }
        }
    }
}

fn is_block(expr: &Expr) -> bool {
    if let ExprKind::Block(..) = expr.kind {
        true
    } else {
        false
    }
}

/// Check if the expression is an `if` or `if let`
fn is_if(expr: &Expr) -> bool {
    if let ExprKind::If(..) = expr.kind {
        true
    } else {
        false
    }
}
