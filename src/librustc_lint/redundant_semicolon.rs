use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_errors::Applicability;
use syntax::ast::{ExprKind, Stmt, StmtKind};

declare_lint! {
    pub REDUNDANT_SEMICOLON,
    Warn,
    "detects unnecessary trailing semicolons"
}

declare_lint_pass!(RedundantSemicolon => [REDUNDANT_SEMICOLON]);

impl EarlyLintPass for RedundantSemicolon {
    fn check_stmt(&mut self, cx: &EarlyContext<'_>, stmt: &Stmt) {
        if let StmtKind::Semi(expr) = &stmt.kind {
            if let ExprKind::Tup(ref v) = &expr.kind {
                if v.is_empty() {
                    // Strings of excess semicolons are encoded as empty tuple expressions
                    // during the parsing stage, so we check for empty tuple expressions
                    // which span only semicolons
                    if let Ok(source_str) = cx.sess().source_map().span_to_snippet(stmt.span) {
                        if source_str.chars().all(|c| c == ';') {
                            let multiple = (stmt.span.hi() - stmt.span.lo()).0 > 1;
                            let msg = if multiple {
                                "unnecessary trailing semicolons"
                            } else {
                                "unnecessary trailing semicolon"
                            };
                            cx.struct_span_lint(REDUNDANT_SEMICOLON, stmt.span, |lint| {
                                let mut err = lint.build(&msg);
                                let suggest_msg = if multiple {
                                    "remove these semicolons"
                                } else {
                                    "remove this semicolon"
                                };
                                err.span_suggestion(
                                    stmt.span,
                                    &suggest_msg,
                                    String::new(),
                                    Applicability::MaybeIncorrect,
                                );
                                err.emit();
                            });
                        }
                    }
                }
            }
        }
    }
}
