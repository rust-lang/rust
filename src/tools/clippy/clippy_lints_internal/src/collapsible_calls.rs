use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::{SpanlessEq, is_expr_path_def_path, is_lint_allowed, peel_blocks_with_stmt};
use rustc_errors::Applicability;
use rustc_hir::{Closure, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::declare_tool_lint;
use rustc_session::declare_lint_pass;
use rustc_span::Span;

use std::borrow::{Borrow, Cow};

declare_tool_lint! {
    /// ### What it does
    /// Lints `span_lint_and_then` function calls, where the
    /// closure argument has only one statement and that statement is a method
    /// call to `span_suggestion`, `span_help`, `span_note` (using the same
    /// span), `help` or `note`.
    ///
    /// These usages of `span_lint_and_then` should be replaced with one of the
    /// wrapper functions `span_lint_and_sugg`, span_lint_and_help`, or
    /// `span_lint_and_note`.
    ///
    /// ### Why is this bad?
    /// Using the wrapper `span_lint_and_*` functions, is more
    /// convenient, readable and less error prone.
    ///
    /// ### Example
    /// ```rust,ignore
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.span_suggestion(
    ///         expr.span,
    ///         help_msg,
    ///         sugg.to_string(),
    ///         Applicability::MachineApplicable,
    ///     );
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.span_help(expr.span, help_msg);
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.help(help_msg);
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.span_note(expr.span, note_msg);
    /// });
    /// span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |diag| {
    ///     diag.note(note_msg);
    /// });
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// span_lint_and_sugg(
    ///     cx,
    ///     TEST_LINT,
    ///     expr.span,
    ///     lint_msg,
    ///     help_msg,
    ///     sugg.to_string(),
    ///     Applicability::MachineApplicable,
    /// );
    /// span_lint_and_help(cx, TEST_LINT, expr.span, lint_msg, Some(expr.span), help_msg);
    /// span_lint_and_help(cx, TEST_LINT, expr.span, lint_msg, None, help_msg);
    /// span_lint_and_note(cx, TEST_LINT, expr.span, lint_msg, Some(expr.span), note_msg);
    /// span_lint_and_note(cx, TEST_LINT, expr.span, lint_msg, None, note_msg);
    /// ```
    pub clippy::COLLAPSIBLE_SPAN_LINT_CALLS,
    Warn,
    "found collapsible `span_lint_and_then` calls",
    report_in_external_macro: true
}

declare_lint_pass!(CollapsibleCalls => [COLLAPSIBLE_SPAN_LINT_CALLS]);

impl<'tcx> LateLintPass<'tcx> for CollapsibleCalls {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if is_lint_allowed(cx, COLLAPSIBLE_SPAN_LINT_CALLS, expr.hir_id) {
            return;
        }

        if let ExprKind::Call(func, [call_cx, call_lint, call_sp, call_msg, call_f]) = expr.kind
            && is_expr_path_def_path(cx, func, &["clippy_utils", "diagnostics", "span_lint_and_then"])
            && let ExprKind::Closure(&Closure { body, .. }) = call_f.kind
            && let body = cx.tcx.hir_body(body)
            && let only_expr = peel_blocks_with_stmt(body.value)
            && let ExprKind::MethodCall(ps, recv, span_call_args, _) = &only_expr.kind
            && let ExprKind::Path(..) = recv.kind
        {
            let and_then_snippets =
                get_and_then_snippets(cx, call_cx.span, call_lint.span, call_sp.span, call_msg.span);
            let mut sle = SpanlessEq::new(cx).deny_side_effects();
            match ps.ident.as_str() {
                "span_suggestion" if sle.eq_expr(call_sp, &span_call_args[0]) => {
                    suggest_suggestion(
                        cx,
                        expr,
                        &and_then_snippets,
                        &span_suggestion_snippets(cx, span_call_args),
                    );
                },
                "span_help" if sle.eq_expr(call_sp, &span_call_args[0]) => {
                    let help_snippet = snippet(cx, span_call_args[1].span, r#""...""#);
                    suggest_help(cx, expr, &and_then_snippets, help_snippet.borrow(), true);
                },
                "span_note" if sle.eq_expr(call_sp, &span_call_args[0]) => {
                    let note_snippet = snippet(cx, span_call_args[1].span, r#""...""#);
                    suggest_note(cx, expr, &and_then_snippets, note_snippet.borrow(), true);
                },
                "help" => {
                    let help_snippet = snippet(cx, span_call_args[0].span, r#""...""#);
                    suggest_help(cx, expr, &and_then_snippets, help_snippet.borrow(), false);
                },
                "note" => {
                    let note_snippet = snippet(cx, span_call_args[0].span, r#""...""#);
                    suggest_note(cx, expr, &and_then_snippets, note_snippet.borrow(), false);
                },
                _ => (),
            }
        }
    }
}

struct AndThenSnippets<'a> {
    cx: Cow<'a, str>,
    lint: Cow<'a, str>,
    span: Cow<'a, str>,
    msg: Cow<'a, str>,
}

fn get_and_then_snippets(
    cx: &LateContext<'_>,
    cx_span: Span,
    lint_span: Span,
    span_span: Span,
    msg_span: Span,
) -> AndThenSnippets<'static> {
    let cx_snippet = snippet(cx, cx_span, "cx");
    let lint_snippet = snippet(cx, lint_span, "..");
    let span_snippet = snippet(cx, span_span, "span");
    let msg_snippet = snippet(cx, msg_span, r#""...""#);

    AndThenSnippets {
        cx: cx_snippet,
        lint: lint_snippet,
        span: span_snippet,
        msg: msg_snippet,
    }
}

struct SpanSuggestionSnippets<'a> {
    help: Cow<'a, str>,
    sugg: Cow<'a, str>,
    applicability: Cow<'a, str>,
}

fn span_suggestion_snippets<'a, 'hir>(
    cx: &LateContext<'_>,
    span_call_args: &'hir [Expr<'hir>],
) -> SpanSuggestionSnippets<'a> {
    let help_snippet = snippet(cx, span_call_args[1].span, r#""...""#);
    let sugg_snippet = snippet(cx, span_call_args[2].span, "..");
    let applicability_snippet = snippet(cx, span_call_args[3].span, "Applicability::MachineApplicable");

    SpanSuggestionSnippets {
        help: help_snippet,
        sugg: sugg_snippet,
        applicability: applicability_snippet,
    }
}

fn suggest_suggestion(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    and_then_snippets: &AndThenSnippets<'_>,
    span_suggestion_snippets: &SpanSuggestionSnippets<'_>,
) {
    span_lint_and_sugg(
        cx,
        COLLAPSIBLE_SPAN_LINT_CALLS,
        expr.span,
        "this call is collapsible",
        "collapse into",
        format!(
            "span_lint_and_sugg({}, {}, {}, {}, {}, {}, {})",
            and_then_snippets.cx,
            and_then_snippets.lint,
            and_then_snippets.span,
            and_then_snippets.msg,
            span_suggestion_snippets.help,
            span_suggestion_snippets.sugg,
            span_suggestion_snippets.applicability
        ),
        Applicability::MachineApplicable,
    );
}

fn suggest_help(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    and_then_snippets: &AndThenSnippets<'_>,
    help: &str,
    with_span: bool,
) {
    let option_span = if with_span {
        format!("Some({})", and_then_snippets.span)
    } else {
        "None".to_string()
    };

    span_lint_and_sugg(
        cx,
        COLLAPSIBLE_SPAN_LINT_CALLS,
        expr.span,
        "this call is collapsible",
        "collapse into",
        format!(
            "span_lint_and_help({}, {}, {}, {}, {}, {help})",
            and_then_snippets.cx, and_then_snippets.lint, and_then_snippets.span, and_then_snippets.msg, &option_span,
        ),
        Applicability::MachineApplicable,
    );
}

fn suggest_note(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    and_then_snippets: &AndThenSnippets<'_>,
    note: &str,
    with_span: bool,
) {
    let note_span = if with_span {
        format!("Some({})", and_then_snippets.span)
    } else {
        "None".to_string()
    };

    span_lint_and_sugg(
        cx,
        COLLAPSIBLE_SPAN_LINT_CALLS,
        expr.span,
        "this call is collapsible",
        "collapse into",
        format!(
            "span_lint_and_note({}, {}, {}, {}, {note_span}, {note})",
            and_then_snippets.cx, and_then_snippets.lint, and_then_snippets.span, and_then_snippets.msg,
        ),
        Applicability::MachineApplicable,
    );
}
