use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{
    HasSession, IntoSpan as _, SpanRangeExt, snippet, snippet_block, snippet_block_with_applicability,
};
use clippy_utils::span_contains_comment;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for nested `if` statements which can be collapsed
    /// by `&&`-combining their conditions.
    ///
    /// ### Why is this bad?
    /// Each `if`-statement adds one level of nesting, which
    /// makes code look more complex than it really is.
    ///
    /// ### Example
    /// ```no_run
    /// # let (x, y) = (true, true);
    /// if x {
    ///     if y {
    ///         // …
    ///     }
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # let (x, y) = (true, true);
    /// if x && y {
    ///     // …
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub COLLAPSIBLE_IF,
    style,
    "nested `if`s that can be collapsed (e.g., `if x { if y { ... } }`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for collapsible `else { if ... }` expressions
    /// that can be collapsed to `else if ...`.
    ///
    /// ### Why is this bad?
    /// Each `if`-statement adds one level of nesting, which
    /// makes code look more complex than it really is.
    ///
    /// ### Example
    /// ```rust,ignore
    ///
    /// if x {
    ///     …
    /// } else {
    ///     if y {
    ///         …
    ///     }
    /// }
    /// ```
    ///
    /// Should be written:
    ///
    /// ```rust,ignore
    /// if x {
    ///     …
    /// } else if y {
    ///     …
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub COLLAPSIBLE_ELSE_IF,
    style,
    "nested `else`-`if` expressions that can be collapsed (e.g., `else { if x { ... } }`)"
}

pub struct CollapsibleIf {
    lint_commented_code: bool,
}

impl CollapsibleIf {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            lint_commented_code: conf.lint_commented_code,
        }
    }

    fn check_collapsible_else_if(cx: &EarlyContext<'_>, then_span: Span, else_: &ast::Expr) {
        if let ast::ExprKind::Block(ref block, _) = else_.kind
            && !block_starts_with_comment(cx, block)
            && let Some(else_) = expr_block(block)
            && else_.attrs.is_empty()
            && !else_.span.from_expansion()
            && let ast::ExprKind::If(..) = else_.kind
        {
            // Prevent "elseif"
            // Check that the "else" is followed by whitespace
            let up_to_else = then_span.between(block.span);
            let requires_space = if let Some(c) = snippet(cx, up_to_else, "..").chars().last() {
                !c.is_whitespace()
            } else {
                false
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                COLLAPSIBLE_ELSE_IF,
                block.span,
                "this `else { if .. }` block can be collapsed",
                "collapse nested if block",
                format!(
                    "{}{}",
                    if requires_space { " " } else { "" },
                    snippet_block_with_applicability(cx, else_.span, "..", Some(block.span), &mut applicability)
                ),
                applicability,
            );
        }
    }

    fn check_collapsible_if_if(&self, cx: &EarlyContext<'_>, expr: &ast::Expr, check: &ast::Expr, then: &ast::Block) {
        if let Some(inner) = expr_block(then)
            && inner.attrs.is_empty()
            && let ast::ExprKind::If(check_inner, _, None) = &inner.kind
            // Prevent triggering on `if c { if let a = b { .. } }`.
            && !matches!(check_inner.kind, ast::ExprKind::Let(..))
            && let ctxt = expr.span.ctxt()
            && inner.span.ctxt() == ctxt
            && let contains_comment = span_contains_comment(cx.sess().source_map(), check.span.to(check_inner.span))
            && (!contains_comment || self.lint_commented_code)
        {
            span_lint_and_then(
                cx,
                COLLAPSIBLE_IF,
                expr.span,
                "this `if` statement can be collapsed",
                |diag| {
                    let then_open_bracket = then.span.split_at(1).0.with_leading_whitespace(cx).into_span();
                    let then_closing_bracket = {
                        let end = then.span.shrink_to_hi();
                        end.with_lo(end.lo() - rustc_span::BytePos(1))
                            .with_leading_whitespace(cx)
                            .into_span()
                    };
                    let inner_if = inner.span.split_at(2).0;
                    let mut sugg = vec![
                        // Remove the outer then block `{`
                        (then_open_bracket, String::new()),
                        // Remove the outer then block '}'
                        (then_closing_bracket, String::new()),
                        // Replace inner `if` by `&&`
                        (inner_if, String::from("&&")),
                    ];
                    sugg.extend(parens_around(check));
                    sugg.extend(parens_around(check_inner));

                    diag.multipart_suggestion("collapse nested if block", sugg, Applicability::MachineApplicable);
                },
            );
        }
    }
}

impl_lint_pass!(CollapsibleIf => [COLLAPSIBLE_IF, COLLAPSIBLE_ELSE_IF]);

impl EarlyLintPass for CollapsibleIf {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, expr: &ast::Expr) {
        if let ast::ExprKind::If(cond, then, else_) = &expr.kind
            && !expr.span.from_expansion()
        {
            if let Some(else_) = else_ {
                Self::check_collapsible_else_if(cx, then.span, else_);
            } else if !matches!(cond.kind, ast::ExprKind::Let(..)) {
                // Prevent triggering on `if c { if let a = b { .. } }`.
                self.check_collapsible_if_if(cx, expr, cond, then);
            }
        }
    }
}

fn block_starts_with_comment(cx: &EarlyContext<'_>, expr: &ast::Block) -> bool {
    // We trim all opening braces and whitespaces and then check if the next string is a comment.
    let trimmed_block_text = snippet_block(cx, expr.span, "..", None)
        .trim_start_matches(|c: char| c.is_whitespace() || c == '{')
        .to_owned();
    trimmed_block_text.starts_with("//") || trimmed_block_text.starts_with("/*")
}

/// If the block contains only one expression, return it.
fn expr_block(block: &ast::Block) -> Option<&ast::Expr> {
    if let [stmt] = &*block.stmts
        && let ast::StmtKind::Expr(expr) | ast::StmtKind::Semi(expr) = &stmt.kind
    {
        Some(expr)
    } else {
        None
    }
}

/// If the expression is a `||`, suggest parentheses around it.
fn parens_around(expr: &ast::Expr) -> Vec<(Span, String)> {
    if let ast::ExprKind::Binary(op, _, _) = expr.kind
        && op.node == ast::BinOpKind::Or
    {
        vec![
            (expr.span.shrink_to_lo(), String::from("(")),
            (expr.span.shrink_to_hi(), String::from(")")),
        ]
    } else {
        vec![]
    }
}
