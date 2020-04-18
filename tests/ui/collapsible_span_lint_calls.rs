// run-rustfix
#![deny(clippy::internal)]
#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_errors;
extern crate rustc_lint;
extern crate rustc_session;
extern crate rustc_span;

use rustc_ast::ast::Expr;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_lint::{EarlyContext, EarlyLintPass, Lint, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

#[allow(unused_variables)]
pub fn span_lint_and_then<'a, T: LintContext, F>(cx: &'a T, lint: &'static Lint, sp: Span, msg: &str, f: F)
where
    F: for<'b> FnOnce(&mut DiagnosticBuilder<'b>),
{
}

#[allow(unused_variables)]
fn span_lint_and_help<'a, T: LintContext>(
    cx: &'a T,
    lint: &'static Lint,
    span: Span,
    msg: &str,
    option_span: Option<Span>,
    help: &str,
) {
}

#[allow(unused_variables)]
fn span_lint_and_note<'a, T: LintContext>(
    cx: &'a T,
    lint: &'static Lint,
    span: Span,
    msg: &str,
    note_span: Option<Span>,
    note: &str,
) {
}

#[allow(unused_variables)]
fn span_lint_and_sugg<'a, T: LintContext>(
    cx: &'a T,
    lint: &'static Lint,
    sp: Span,
    msg: &str,
    help: &str,
    sugg: String,
    applicability: Applicability,
) {
}

declare_tool_lint! {
    pub clippy::TEST_LINT,
    Warn,
    "",
    report_in_external_macro: true
}

declare_lint_pass!(Pass => [TEST_LINT]);

impl EarlyLintPass for Pass {
    fn check_expr(&mut self, cx: &EarlyContext, expr: &Expr) {
        let lint_msg = "lint message";
        let help_msg = "help message";
        let note_msg = "note message";
        let sugg = "new_call()";
        let predicate = true;

        span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |db| {
            db.span_suggestion(expr.span, help_msg, sugg.to_string(), Applicability::MachineApplicable);
        });
        span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |db| {
            db.span_help(expr.span, help_msg);
        });
        span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |db| {
            db.help(help_msg);
        });
        span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |db| {
            db.span_note(expr.span, note_msg);
        });
        span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |db| {
            db.note(note_msg);
        });

        // This expr shouldn't trigger this lint.
        span_lint_and_then(cx, TEST_LINT, expr.span, lint_msg, |db| {
            db.note(note_msg);
            if predicate {
                db.note(note_msg);
            }
        })
    }
}

fn main() {}
