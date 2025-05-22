use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{IntoSpan as _, SpanRangeExt, snippet, snippet_block, snippet_block_with_applicability};
use rustc_ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
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
    msrv: Msrv,
    lint_commented_code: bool,
}

impl CollapsibleIf {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            lint_commented_code: conf.lint_commented_code,
        }
    }

    fn check_collapsible_else_if(cx: &LateContext<'_>, then_span: Span, else_block: &Block<'_>) {
        if !block_starts_with_comment(cx, else_block)
            && let Some(else_) = expr_block(else_block)
            && cx.tcx.hir_attrs(else_.hir_id).is_empty()
            && !else_.span.from_expansion()
            && let ExprKind::If(..) = else_.kind
        {
            // Prevent "elseif"
            // Check that the "else" is followed by whitespace
            let up_to_else = then_span.between(else_block.span);
            let requires_space = if let Some(c) = snippet(cx, up_to_else, "..").chars().last() {
                !c.is_whitespace()
            } else {
                false
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                COLLAPSIBLE_ELSE_IF,
                else_block.span,
                "this `else { if .. }` block can be collapsed",
                "collapse nested if block",
                format!(
                    "{}{}",
                    if requires_space { " " } else { "" },
                    snippet_block_with_applicability(cx, else_.span, "..", Some(else_block.span), &mut applicability)
                ),
                applicability,
            );
        }
    }

    fn check_collapsible_if_if(&self, cx: &LateContext<'_>, expr: &Expr<'_>, check: &Expr<'_>, then: &Block<'_>) {
        if let Some(inner) = expr_block(then)
            && cx.tcx.hir_attrs(inner.hir_id).is_empty()
            && let ExprKind::If(check_inner, _, None) = &inner.kind
            && self.eligible_condition(cx, check_inner)
            && let ctxt = expr.span.ctxt()
            && inner.span.ctxt() == ctxt
            && (self.lint_commented_code || !block_starts_with_comment(cx, then))
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

    fn eligible_condition(&self, cx: &LateContext<'_>, cond: &Expr<'_>) -> bool {
        !matches!(cond.kind, ExprKind::Let(..))
            || (cx.tcx.sess.edition().at_least_rust_2024() && self.msrv.meets(cx, msrvs::LET_CHAINS))
    }
}

impl_lint_pass!(CollapsibleIf => [COLLAPSIBLE_IF, COLLAPSIBLE_ELSE_IF]);

impl LateLintPass<'_> for CollapsibleIf {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::If(cond, then, else_) = &expr.kind
            && !expr.span.from_expansion()
        {
            if let Some(else_) = else_
                && let ExprKind::Block(else_, None) = else_.kind
            {
                Self::check_collapsible_else_if(cx, then.span, else_);
            } else if else_.is_none()
                && self.eligible_condition(cx, cond)
                && let ExprKind::Block(then, None) = then.kind
            {
                self.check_collapsible_if_if(cx, expr, cond, then);
            }
        }
    }
}

fn block_starts_with_comment(cx: &LateContext<'_>, block: &Block<'_>) -> bool {
    // We trim all opening braces and whitespaces and then check if the next string is a comment.
    let trimmed_block_text = snippet_block(cx, block.span, "..", None)
        .trim_start_matches(|c: char| c.is_whitespace() || c == '{')
        .to_owned();
    trimmed_block_text.starts_with("//") || trimmed_block_text.starts_with("/*")
}

/// If `block` is a block with either one expression or a statement containing an expression,
/// return the expression. We don't peel blocks recursively, as extra blocks might be intentional.
fn expr_block<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match block.stmts {
        [] => block.expr,
        [
            Stmt {
                kind: StmtKind::Semi(expr),
                ..
            },
        ] if block.expr.is_none() => Some(expr),
        _ => None,
    }
}

/// If the expression is a `||`, suggest parentheses around it.
fn parens_around(expr: &Expr<'_>) -> Vec<(Span, String)> {
    if let ExprKind::Binary(op, _, _) = expr.peel_drop_temps().kind
        && op.node == BinOpKind::Or
    {
        vec![
            (expr.span.shrink_to_lo(), String::from("(")),
            (expr.span.shrink_to_hi(), String::from(")")),
        ]
    } else {
        vec![]
    }
}
