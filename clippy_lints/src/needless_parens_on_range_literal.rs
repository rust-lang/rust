//! Checks for parantheses on literals in range statements
//!
//! For example, the lint would catch
//!
//! ```rust
//! for i in (0)..10 {
//!   println!("{i}");
//! }
//! ```
//!
//! Use instead:
//!
//! ```rust
//! for i in 0..10 {
//!   println!("{i}");
//! }
//! ```
//!

use clippy_utils::{diagnostics::span_lint_and_then, source::snippet_opt};
use rustc_ast::ast::{Expr, ExprKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
  /// ### What it does
  /// The lint checks for parenthesis on literals in range statements that are
  /// superflous.
  ///
  /// ### Why is this bad?
  /// Having superflous parenthesis makes the code less legible as the impose an
  /// overhead when reading.

  #[clippy::version = "1.63.0"]
  pub NEEDLESS_PARENS_ON_RANGE_LITERAL,
  style,
  "needless parenthesis on range literal can be removed"
}

declare_lint_pass!(NeedlessParensOnRangeLiteral => [NEEDLESS_PARENS_ON_RANGE_LITERAL]);

fn check_for_parens(cx: &EarlyContext<'_>, e: &Expr) {
    if_chain! {
      if let ExprKind::Paren(ref start_statement) = &e.kind;
    if let ExprKind::Lit(ref literal) = start_statement.kind;
    then {
      span_lint_and_then(cx, NEEDLESS_PARENS_ON_RANGE_LITERAL, e.span,
        "needless parenthesis on range literal can be removed",
        |diag| {
                if let Some(suggestion) = snippet_opt(cx, literal.span) {
                  diag.span_suggestion(e.span, "try", suggestion, Applicability::MachineApplicable);
              }
        });
    }
    }
}

impl EarlyLintPass for NeedlessParensOnRangeLiteral {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &Expr) {
        if let ExprKind::Range(Some(start), Some(end), _) = &e.kind {
            check_for_parens(cx, start);
            check_for_parens(cx, end);
        }
    }
}
