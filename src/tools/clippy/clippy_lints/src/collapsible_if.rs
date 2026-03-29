use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::msrvs::Msrv;
use clippy_utils::source::{HasSession, IntoSpan as _, SpanRangeExt, snippet, snippet_block_with_applicability};
use clippy_utils::{can_use_if_let_chains, span_contains_non_whitespace, sym, tokenize_with_text};
use rustc_ast::BinOpKind;
use rustc_errors::Applicability;
use rustc_hir::attrs::{AttributeKind, LintAttributeKind};
use rustc_hir::{Attribute, Block, Expr, ExprKind, StmtKind};
use rustc_lexer::TokenKind;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{BytePos, Span, Symbol};

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
    pedantic,
    "nested `else`-`if` expressions that can be collapsed (e.g., `else { if x { ... } }`)"
}

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

impl_lint_pass!(CollapsibleIf => [COLLAPSIBLE_ELSE_IF, COLLAPSIBLE_IF]);

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
            && !else_.span.from_expansion()
            && let ExprKind::If(else_if_cond, ..) = else_.kind
            && self.check_significant_tokens_and_expect_attrs(cx, else_block, else_, sym::collapsible_else_if)
        {
            span_lint_hir_and_then(
                cx,
                COLLAPSIBLE_ELSE_IF,
                else_.hir_id,
                else_block.span,
                "this `else { if .. }` block can be collapsed",
                |diag| {
                    let up_to_else = then_span.between(else_block.span);
                    let else_before_if = else_.span.shrink_to_lo().with_hi(else_if_cond.span.lo() - BytePos(1));
                    if self.lint_commented_code
                        && let Some(else_keyword_span) = span_extract_keyword(cx, up_to_else, "else")
                        && let Some(else_if_keyword_span) = span_extract_keyword(cx, else_before_if, "if")
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

                    // Peel off any parentheses.
                    let (_, else_block_span, _) = peel_parens(cx, else_.span);

                    // Prevent "elseif"
                    // Check that the "else" is followed by whitespace
                    let requires_space = snippet(cx, up_to_else, "..").ends_with(|c: char| !c.is_whitespace());
                    let mut applicability = Applicability::MachineApplicable;
                    diag.span_suggestion(
                        else_block.span,
                        "collapse nested if block",
                        format!(
                            "{}{}",
                            if requires_space { " " } else { "" },
                            snippet_block_with_applicability(
                                cx,
                                else_block_span,
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
            && let ExprKind::If(check_inner, _, None) = &inner.kind
            && self.eligible_condition(cx, check_inner)
            && expr.span.eq_ctxt(inner.span)
            && self.check_significant_tokens_and_expect_attrs(cx, then, inner, sym::collapsible_if)
        {
            span_lint_hir_and_then(
                cx,
                COLLAPSIBLE_IF,
                inner.hir_id,
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
                    let (paren_start, inner_if_span, paren_end) = peel_parens(cx, inner.span);
                    let inner_if = inner_if_span.split_at(2).0;
                    let mut sugg = vec![
                        // Remove the outer then block `{`
                        (then_open_bracket, String::new()),
                        // Remove the outer then block '}'
                        (then_closing_bracket, String::new()),
                        // Replace inner `if` by `&&`
                        (inner_if, String::from("&&")),
                    ];

                    if !paren_start.is_empty() {
                        // Remove any leading parentheses '('
                        sugg.push((paren_start, String::new()));
                    }

                    if !paren_end.is_empty() {
                        // Remove any trailing parentheses ')'
                        sugg.push((paren_end, String::new()));
                    }

                    sugg.extend(parens_around(check));
                    sugg.extend(parens_around(check_inner));

                    diag.multipart_suggestion("collapse nested if block", sugg, Applicability::MachineApplicable);
                },
            );
        }
    }

    fn eligible_condition(&self, cx: &LateContext<'_>, cond: &Expr<'_>) -> bool {
        !matches!(cond.kind, ExprKind::Let(..)) || can_use_if_let_chains(cx, self.msrv)
    }

    // Check that nothing significant can be found between the initial `{` of `inner_if` and
    // the beginning of `inner_if_expr`...
    //
    // Unless it's only an `#[expect(clippy::collapsible{,_else}_if)]` attribute, in which case we
    // _do_ need to lint, in order to actually fulfill its expectation (#13365)
    fn check_significant_tokens_and_expect_attrs(
        &self,
        cx: &LateContext<'_>,
        inner_if: &Block<'_>,
        inner_if_expr: &Expr<'_>,
        expected_lint_name: Symbol,
    ) -> bool {
        match cx.tcx.hir_attrs(inner_if_expr.hir_id) {
            [] => {
                // There aren't any attributes, so just check for significant tokens
                let span = inner_if.span.split_at(1).1.until(inner_if_expr.span);
                !span_contains_non_whitespace(cx, span, self.lint_commented_code)
            },

            [
                Attribute::Parsed(AttributeKind::LintAttributes(sub_attrs)),
            ] => {
                sub_attrs
                    .into_iter()
                    .filter(|attr|attr.kind == LintAttributeKind::Expect)
                    .flat_map(|attr| attr.lint_instances.iter().map(|group| (attr.attr_span, group)))
                    .filter(|(_, lint_id)| {
                        lint_id.tool_is_named(sym::clippy)
                            && (expected_lint_name == lint_id.lint_name()
                                || [expected_lint_name, sym::style, sym::all]
                                    .contains(&lint_id.original_name_without_tool()))
                    })
                    .any(|(attr_span, _)| {
                        // There is an `expect` attribute -- check that there is no _other_ significant text
                        let span_before_attr = inner_if.span.split_at(1).1.until(attr_span);
                        let span_after_attr = attr_span.between(inner_if_expr.span);
                        !span_contains_non_whitespace(cx, span_before_attr, self.lint_commented_code)
                            && !span_contains_non_whitespace(cx, span_after_attr, self.lint_commented_code)
                    })
            },

            // There are other attributes, which are significant tokens -- check failed
            _ => false,
        }
    }
}

impl LateLintPass<'_> for CollapsibleIf {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::If(cond, then, else_) = &expr.kind
            && !expr.span.from_expansion()
        {
            if let Some(else_) = else_
                // Short circuit if both `if` branches contain only a single `if {..} else {}`, as
                // collapsing such blocks can lead to less readable code (#4971)
                && !(single_inner_if_else(then) && single_inner_if_else(else_))
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

/// Returns true if `expr` is a block that contains only one `if {..} else {}` statement
fn single_inner_if_else(expr: &Expr<'_>) -> bool {
    if let ExprKind::Block(block, None) = expr.kind
        && let Some(inner_expr) = expr_block(block)
        && let ExprKind::If(_, _, else_) = inner_expr.kind
        && else_.is_some()
    {
        true
    } else {
        false
    }
}

/// If `block` is a block with either one expression or a statement containing an expression,
/// return the expression. We don't peel blocks recursively, as extra blocks might be intentional.
fn expr_block<'tcx>(block: &Block<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    match (block.stmts, block.expr) {
        ([], expr) => expr,
        ([stmt], None) if let StmtKind::Semi(expr) = stmt.kind => Some(expr),
        _ => None,
    }
}

/// If the expression is a `||`, suggest parentheses around it.
pub(super) fn parens_around(expr: &Expr<'_>) -> Vec<(Span, String)> {
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

fn span_extract_keyword(cx: &impl HasSession, span: Span, keyword: &str) -> Option<Span> {
    span.with_source_text(cx, |snippet| {
        tokenize_with_text(snippet)
            .filter(|(t, s, _)| matches!(t, TokenKind::Ident if *s == keyword))
            .map(|(_, _, inner)| {
                span.split_at(u32::try_from(inner.start).unwrap())
                    .1
                    .split_at(u32::try_from(inner.end - inner.start).unwrap())
                    .0
            })
            .next()
    })
    .flatten()
}

/// Peel the parentheses from an `if` expression, e.g. `((if true {} else {}))`.
pub(super) fn peel_parens(cx: &impl HasSession, mut span: Span) -> (Span, Span, Span) {
    use crate::rustc_span::Pos;

    let start = span.shrink_to_lo();
    let end = span.shrink_to_hi();

    span.with_source_text(cx, |snippet| {
        if let Some((trim_start, _, trim_end)) = peel_parens_str(snippet) {
            let mut data = span.data();
            data.lo = data.lo + BytePos::from_usize(trim_start);
            data.hi = data.hi - BytePos::from_usize(trim_end);
            span = data.span();
        }
    });

    (start.with_hi(span.lo()), span, end.with_lo(span.hi()))
}

fn peel_parens_str(snippet: &str) -> Option<(usize, &str, usize)> {
    let trimmed = snippet.trim();
    if !(trimmed.starts_with('(') && trimmed.ends_with(')')) {
        return None;
    }

    let trim_start = (snippet.len() - snippet.trim_start().len()) + 1;
    let trim_end = (snippet.len() - snippet.trim_end().len()) + 1;

    let inner = snippet.get(trim_start..snippet.len() - trim_end)?;
    Some(match peel_parens_str(inner) {
        None => (trim_start, inner, trim_end),
        Some((start, inner, end)) => (trim_start + start, inner, trim_end + end),
    })
}
