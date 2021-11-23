use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_macro_callsite;
use rustc_errors::Applicability;
use rustc_hir::{Stmt, StmtKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;

use super::LET_UNIT_VALUE;

pub(super) fn check(cx: &LateContext<'_>, stmt: &Stmt<'_>) {
    if let StmtKind::Local(local) = stmt.kind {
        if cx.typeck_results().pat_ty(local.pat).is_unit() {
            if in_external_macro(cx.sess(), stmt.span) || local.pat.span.from_expansion() {
                return;
            }
            span_lint_and_then(
                cx,
                LET_UNIT_VALUE,
                stmt.span,
                "this let-binding has unit value",
                |diag| {
                    if let Some(expr) = &local.init {
                        let snip = snippet_with_macro_callsite(cx, expr.span, "()");
                        diag.span_suggestion(
                            stmt.span,
                            "omit the `let` binding",
                            format!("{};", snip),
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                },
            );
        }
    }
}
