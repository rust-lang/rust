use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{IntoSpan as _, SpanRangeExt, snippet, snippet_block_with_applicability};
use clippy_utils::{span_contains_non_whitespace, tokenize_with_text};
use rustc_ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::{Block, Expr, ExprKind, Stmt, StmtKind};
use rustc_lexer::TokenKind;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::source_map::SourceMap;
use rustc_span::{BytePos, Span};

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

    fn check_collapsible_else_if(&self, cx: &LateContext<'_>, then_span: Span, else_block: &Block<'_>) {
        if let Some(else_) = expr_block(else_block)
            && cx.tcx.hir_attrs(else_.hir_id).is_empty()
            && !else_.span.from_expansion()
            && let ExprKind::If(else_if_cond, ..) = else_.kind
            && !block_starts_with_significant_tokens(cx, else_block, else_, self.lint_commented_code)
        {
            span_lint_and_then(
                cx,
                COLLAPSIBLE_ELSE_IF,
                else_block.span,
                "this `else { if .. }` block can be collapsed",
                |diag| {
                    let up_to_else = then_span.between(else_block.span);
                    let else_before_if = else_.span.shrink_to_lo().with_hi(else_if_cond.span.lo() - BytePos(1));
                    if self.lint_commented_code
                        && let Some(else_keyword_span) =
                            span_extract_keyword(cx.tcx.sess.source_map(), up_to_else, "else")
                        && let Some(else_if_keyword_span) =
                            span_extract_keyword(cx.tcx.sess.source_map(), else_before_if, "if")
                    {
                        let else_keyword_span = else_keyword_span.with_leading_whitespace(cx).into_span();
                        let else_open_bracket = else_block.span.split_at(1).0.with_leading_whitespace(cx).into_span();
                        let else_closing_bracket = {
                            let end = else_block.span.shrink_to_hi();
                            end.with_lo(end.lo() - BytePos(1))
                                .with_leading_whitespace(cx)
                                .into_span()
                        };
                        let sugg = vec![
                            // Remove the outer else block `else`
                            (else_keyword_span, String::new()),
                            // Replace the inner `if` by `else if`
                            (else_if_keyword_span, String::from("else if")),
                            // Remove the outer else block `{`
                            (else_open_bracket, String::new()),
                            // Remove the outer else block '}'
                            (else_closing_bracket, String::new()),
                        ];
                        diag.multipart_suggestion("collapse nested if block", sugg, Applicability::MachineApplicable);
                        return;
                    }

                    // Prevent "elseif"
                    // Check that the "else" is followed by whitespace
                    let requires_space = if let Some(c) = snippet(cx, up_to_else, "..").chars().last() {
                        !c.is_whitespace()
                    } else {
                        false
                    };
                    let mut applicability = Applicability::MachineApplicable;
                    diag.span_suggestion(
                        else_block.span,
                        "collapse nested if block",
                        format!(
                            "{}{}",
                            if requires_space { " " } else { "" },
                            snippet_block_with_applicability(
                                cx,
                                else_.span,
                                "..",
                                Some(else_block.span),
                                &mut applicability
                            )
                        ),
                        applicability,
                    );
                },
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
            && !block_starts_with_significant_tokens(cx, then, inner, self.lint_commented_code)
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
                        end.with_lo(end.lo() - BytePos(1))
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
                self.check_collapsible_else_if(cx, then.span, else_);
            } else if else_.is_none()
                && self.eligible_condition(cx, cond)
                && let ExprKind::Block(then, None) = then.kind
            {
                self.check_collapsible_if_if(cx, expr, cond, then);
            }
        }
    }
}

// Check that nothing significant can be found but whitespaces between the initial `{` of `block`
// and the beginning of `stop_at`.
fn block_starts_with_significant_tokens(
    cx: &LateContext<'_>,
    block: &Block<'_>,
    stop_at: &Expr<'_>,
    lint_commented_code: bool,
) -> bool {
    let span = block.span.split_at(1).1.until(stop_at.span);
    span_contains_non_whitespace(cx, span, lint_commented_code)
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

fn span_extract_keyword(sm: &SourceMap, span: Span, keyword: &str) -> Option<Span> {
    let snippet = sm.span_to_snippet(span).ok()?;
    tokenize_with_text(&snippet)
        .filter(|(t, s, _)| matches!(t, TokenKind::Ident if *s == keyword))
        .map(|(_, _, inner)| {
            span.split_at(u32::try_from(inner.start).unwrap())
                .1
                .split_at(u32::try_from(inner.end - inner.start).unwrap())
                .0
        })
        .next()
}
