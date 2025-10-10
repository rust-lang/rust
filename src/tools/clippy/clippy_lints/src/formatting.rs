use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_note};
use clippy_utils::is_span_if;
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{BinOpKind, Block, Expr, ExprKind, StmtKind};
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of the non-existent `=*`, `=!` and `=-`
    /// operators.
    ///
    /// ### Why is this bad?
    /// This is either a typo of `*=`, `!=` or `-=` or
    /// confusing.
    ///
    /// ### Example
    /// ```rust,ignore
    /// a =- 42; // confusing, should it be `a -= 42` or `a = -42`?
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub SUSPICIOUS_ASSIGNMENT_FORMATTING,
    suspicious,
    "suspicious formatting of `*=`, `-=` or `!=`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks the formatting of a unary operator on the right hand side
    /// of a binary operator. It lints if there is no space between the binary and unary operators,
    /// but there is a space between the unary and its operand.
    ///
    /// ### Why is this bad?
    /// This is either a typo in the binary operator or confusing.
    ///
    /// ### Example
    /// ```no_run
    /// # let foo = true;
    /// # let bar = false;
    /// // &&! looks like a different operator
    /// if foo &&! bar {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let foo = true;
    /// # let bar = false;
    /// if foo && !bar {}
    /// ```
    #[clippy::version = "1.40.0"]
    pub SUSPICIOUS_UNARY_OP_FORMATTING,
    suspicious,
    "suspicious formatting of unary `-` or `!` on the RHS of a BinOp"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for formatting of `else`. It lints if the `else`
    /// is followed immediately by a newline or the `else` seems to be missing.
    ///
    /// ### Why is this bad?
    /// This is probably some refactoring remnant, even if the
    /// code is correct, it might look confusing.
    ///
    /// ### Example
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
    #[clippy::version = "pre 1.29.0"]
    pub SUSPICIOUS_ELSE_FORMATTING,
    suspicious,
    "suspicious formatting of `else`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for an `if` expression followed by either a block or another `if` that
    /// looks like it should have an `else` between them.
    ///
    /// ### Why is this bad?
    /// This is probably some refactoring remnant, even if the code is correct, it
    /// might look confusing.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if foo {
    /// } { // looks like an `else` is missing here
    /// }
    ///
    /// if foo {
    /// } if bar { // looks like an `else` is missing here
    /// }
    /// ```
    #[clippy::version = "1.90.0"]
    pub POSSIBLE_MISSING_ELSE,
    suspicious,
    "possibly missing `else`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for possible missing comma in an array. It lints if
    /// an array element is a binary operator expression and it lies on two lines.
    ///
    /// ### Why is this bad?
    /// This could lead to unexpected results.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a = &[
    ///     -1, -2, -3 // <= no comma here
    ///     -4, -5, -6
    /// ];
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub POSSIBLE_MISSING_COMMA,
    correctness,
    "possible missing comma in array"
}

declare_lint_pass!(Formatting => [
    SUSPICIOUS_ASSIGNMENT_FORMATTING,
    SUSPICIOUS_UNARY_OP_FORMATTING,
    SUSPICIOUS_ELSE_FORMATTING,
    POSSIBLE_MISSING_ELSE,
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
    if let ExprKind::Assign(ref lhs, ref rhs, _) = expr.kind
        && !lhs.span.from_expansion()
        && !rhs.span.from_expansion()
    {
        let eq_span = lhs.span.between(rhs.span);
        if let ExprKind::Unary(op, ref sub_rhs) = rhs.kind
            && let Some(eq_snippet) = snippet_opt(cx, eq_span)
        {
            let op = op.as_str();
            let eqop_span = lhs.span.between(sub_rhs.span);
            if eq_snippet.ends_with('=') {
                span_lint_and_note(
                    cx,
                    SUSPICIOUS_ASSIGNMENT_FORMATTING,
                    eqop_span,
                    format!(
                        "this looks like you are trying to use `.. {op}= ..`, but you \
                                 really are doing `.. = ({op} ..)`"
                    ),
                    None,
                    format!("to remove this lint, use either `{op}=` or `= {op}`"),
                );
            }
        }
    }
}

/// Implementation of the `SUSPICIOUS_UNARY_OP_FORMATTING` lint.
fn check_unop(cx: &EarlyContext<'_>, expr: &Expr) {
    if let ExprKind::Binary(ref binop, ref lhs, ref rhs) = expr.kind
        && !lhs.span.from_expansion() && !rhs.span.from_expansion()
        // span between BinOp LHS and RHS
        && let binop_span = lhs.span.between(rhs.span)
        // if RHS is an UnOp
        && let ExprKind::Unary(op, ref un_rhs) = rhs.kind
        // from UnOp operator to UnOp operand
        && let unop_operand_span = rhs.span.until(un_rhs.span)
        && let Some(binop_snippet) = snippet_opt(cx, binop_span)
        && let Some(unop_operand_snippet) = snippet_opt(cx, unop_operand_span)
        && let binop_str = binop.node.as_str()
        // no space after BinOp operator and space after UnOp operator
        && binop_snippet.ends_with(binop_str) && unop_operand_snippet.ends_with(' ')
    {
        let unop_str = op.as_str();
        let eqop_span = lhs.span.between(un_rhs.span);
        span_lint_and_help(
            cx,
            SUSPICIOUS_UNARY_OP_FORMATTING,
            eqop_span,
            format!(
                "by not having a space between `{binop_str}` and `{unop_str}` it looks like \
                 `{binop_str}{unop_str}` is a single operator"
            ),
            None,
            format!("put a space between `{binop_str}` and `{unop_str}` and remove the space after `{unop_str}`"),
        );
    }
}

/// Implementation of the `SUSPICIOUS_ELSE_FORMATTING` lint for weird `else`.
fn check_else(cx: &EarlyContext<'_>, expr: &Expr) {
    if let ExprKind::If(_, then, Some(else_)) = &expr.kind
        && (is_block(else_) || is_if(else_))
        && !then.span.from_expansion() && !else_.span.from_expansion()
        && !expr.span.in_external_macro(cx.sess().source_map())

        // workaround for rust-lang/rust#43081
        && expr.span.lo().0 != 0 && expr.span.hi().0 != 0

        // this will be a span from the closing ‘}’ of the “then” block (excluding) to
        // the “if” of the “else if” block (excluding)
        && let else_span = then.span.between(else_.span)

        // the snippet should look like " else \n    " with maybe comments anywhere
        // it’s bad when there is a ‘\n’ after the “else”
        && let Some(else_snippet) = snippet_opt(cx, else_span)
        && let Some((pre_else, post_else)) = else_snippet.split_once("else")
        && !else_snippet.contains('/')
        && let Some((_, post_else_post_eol)) = post_else.split_once('\n')
    {
        // Allow allman style braces `} \n else \n {`
        if is_block(else_)
            && let Some((_, pre_else_post_eol)) = pre_else.split_once('\n')
            // Exactly one eol before and after the else
            && !pre_else_post_eol.contains('\n')
            && !post_else_post_eol.contains('\n')
        {
            return;
        }

        // Don't warn if the only thing inside post_else_post_eol is a comment block.
        let trimmed_post_else_post_eol = post_else_post_eol.trim();
        if trimmed_post_else_post_eol.starts_with("/*") && trimmed_post_else_post_eol.ends_with("*/") {
            return;
        }

        let else_desc = if is_if(else_) { "if" } else { "{..}" };
        span_lint_and_note(
            cx,
            SUSPICIOUS_ELSE_FORMATTING,
            else_span,
            format!("this is an `else {else_desc}` but the formatting might hide it"),
            None,
            format!(
                "to remove this lint, remove the `else` or remove the new line between \
                 `else` and `{else_desc}`",
            ),
        );
    }
}

#[must_use]
fn has_unary_equivalent(bin_op: BinOpKind) -> bool {
    // &, *, -
    bin_op == BinOpKind::And || bin_op == BinOpKind::Mul || bin_op == BinOpKind::Sub
}

fn indentation(cx: &EarlyContext<'_>, span: Span) -> usize {
    cx.sess().source_map().lookup_char_pos(span.lo()).col.0
}

/// Implementation of the `POSSIBLE_MISSING_COMMA` lint for array
fn check_array(cx: &EarlyContext<'_>, expr: &Expr) {
    if let ExprKind::Array(ref array) = expr.kind {
        for element in array {
            if let ExprKind::Binary(ref op, ref lhs, _) = element.kind
                && has_unary_equivalent(op.node)
                && lhs.span.eq_ctxt(op.span)
                && let space_span = lhs.span.between(op.span)
                && let Some(space_snippet) = snippet_opt(cx, space_span)
                && let lint_span = lhs.span.with_lo(lhs.span.hi())
                && space_snippet.contains('\n')
                && indentation(cx, op.span) <= indentation(cx, lhs.span)
            {
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

fn check_missing_else(cx: &EarlyContext<'_>, first: &Expr, second: &Expr) {
    if !first.span.from_expansion() && !second.span.from_expansion()
        && matches!(first.kind, ExprKind::If(..))
        && (is_block(second) || is_if(second))

        // Proc-macros can give weird spans. Make sure this is actually an `if`.
        && is_span_if(cx, first.span)

        // If there is a line break between the two expressions, don't lint.
        // If there is a non-whitespace character, this span came from a proc-macro.
        && let else_span = first.span.between(second.span)
        && let Some(else_snippet) = snippet_opt(cx, else_span)
        && !else_snippet.chars().any(|c| c == '\n' || !c.is_whitespace())
    {
        let (looks_like, next_thing) = if is_if(second) {
            ("an `else if`", "the second `if`")
        } else {
            ("an `else {..}`", "the next block")
        };

        span_lint_and_note(
            cx,
            POSSIBLE_MISSING_ELSE,
            else_span,
            format!("this looks like {looks_like} but the `else` is missing"),
            None,
            format!("to remove this lint, add the missing `else` or add a new line before {next_thing}",),
        );
    }
}

fn is_block(expr: &Expr) -> bool {
    matches!(expr.kind, ExprKind::Block(..))
}

/// Check if the expression is an `if` or `if let`
fn is_if(expr: &Expr) -> bool {
    matches!(expr.kind, ExprKind::If(..))
}
