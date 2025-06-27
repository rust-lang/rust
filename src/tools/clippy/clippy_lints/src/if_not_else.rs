use clippy_utils::consts::is_zero_integer_const;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::is_else_clause;
use clippy_utils::source::{HasSession, indent_of, reindent_multiline, snippet};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `!` or `!=` in an if condition with an
    /// else branch.
    ///
    /// ### Why is this bad?
    /// Negations reduce the readability of statements.
    ///
    /// ### Example
    /// ```no_run
    /// # let v: Vec<usize> = vec![];
    /// # fn a() {}
    /// # fn b() {}
    /// if !v.is_empty() {
    ///     a()
    /// } else {
    ///     b()
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```no_run
    /// # let v: Vec<usize> = vec![];
    /// # fn a() {}
    /// # fn b() {}
    /// if v.is_empty() {
    ///     b()
    /// } else {
    ///     a()
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IF_NOT_ELSE,
    pedantic,
    "`if` branches that could be swapped so no negation operation is necessary on the condition"
}

declare_lint_pass!(IfNotElse => [IF_NOT_ELSE]);

impl LateLintPass<'_> for IfNotElse {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &Expr<'_>) {
        if let ExprKind::If(cond, cond_inner, Some(els)) = e.kind
            && let ExprKind::DropTemps(cond) = cond.kind
            && let ExprKind::Block(..) = els.kind
        {
            let (msg, help) = match cond.kind {
                ExprKind::Unary(UnOp::Not, _) => (
                    "unnecessary boolean `not` operation",
                    "remove the `!` and swap the blocks of the `if`/`else`",
                ),
                // Don't lint on `… != 0`, as these are likely to be bit tests.
                // For example, `if foo & 0x0F00 != 0 { … } else { … }` is already in the "proper" order.
                ExprKind::Binary(op, _, rhs) if op.node == BinOpKind::Ne && !is_zero_integer_const(cx, rhs) => (
                    "unnecessary `!=` operation",
                    "change to `==` and swap the blocks of the `if`/`else`",
                ),
                _ => return,
            };

            // `from_expansion` will also catch `while` loops which appear in the HIR as:
            // ```rust
            // loop {
            //     if cond { ... } else { break; }
            // }
            // ```
            if !e.span.from_expansion() && !is_else_clause(cx.tcx, e) {
                match cond.kind {
                    ExprKind::Unary(UnOp::Not, _) | ExprKind::Binary(_, _, _) => span_lint_and_sugg(
                        cx,
                        IF_NOT_ELSE,
                        e.span,
                        msg,
                        "try",
                        make_sugg(cx, &cond.kind, cond_inner.span, els.span, "..", Some(e.span)).to_string(),
                        Applicability::MachineApplicable,
                    ),
                    _ => span_lint_and_help(cx, IF_NOT_ELSE, e.span, msg, None, help),
                }
            }
        }
    }
}

fn make_sugg<'a>(
    sess: &impl HasSession,
    cond_kind: &'a ExprKind<'a>,
    cond_inner: Span,
    els_span: Span,
    default: &'a str,
    indent_relative_to: Option<Span>,
) -> String {
    let cond_inner_snip = snippet(sess, cond_inner, default);
    let els_snip = snippet(sess, els_span, default);
    let indent = indent_relative_to.and_then(|s| indent_of(sess, s));

    let suggestion = match cond_kind {
        ExprKind::Unary(UnOp::Not, cond_rest) => {
            format!(
                "if {} {} else {}",
                snippet(sess, cond_rest.span, default),
                els_snip,
                cond_inner_snip
            )
        },
        ExprKind::Binary(_, lhs, rhs) => {
            let lhs_snip = snippet(sess, lhs.span, default);
            let rhs_snip = snippet(sess, rhs.span, default);

            format!("if {lhs_snip} == {rhs_snip} {els_snip} else {cond_inner_snip}")
        },
        _ => String::new(),
    };

    reindent_multiline(&suggestion, true, indent)
}
