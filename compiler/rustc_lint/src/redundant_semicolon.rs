use crate::{EarlyContext, EarlyLintPass, LintContext};
use rustc_ast::{Block, StmtKind};
use rustc_errors::Applicability;
use rustc_span::Span;

declare_lint! {
    /// The `redundant_semicolons` lint detects unnecessary trailing
    /// semicolons.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let _ = 123;;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Extra semicolons are not needed, and may be removed to avoid confusion
    /// and visual clutter.
    pub REDUNDANT_SEMICOLONS,
    Warn,
    "detects unnecessary trailing semicolons"
}

declare_lint_pass!(RedundantSemicolons => [REDUNDANT_SEMICOLONS]);

impl EarlyLintPass for RedundantSemicolons {
    fn check_block(&mut self, cx: &EarlyContext<'_>, block: &Block) {
        let mut after_item_stmt = false;
        let mut seq = None;
        for stmt in block.stmts.iter() {
            match (&stmt.kind, &mut seq) {
                (StmtKind::Empty, None) => seq = Some((stmt.span, false)),
                (StmtKind::Empty, Some(seq)) => *seq = (seq.0.to(stmt.span), true),
                (_, seq) => {
                    maybe_lint_redundant_semis(cx, seq, after_item_stmt);
                    after_item_stmt = matches!(stmt.kind, StmtKind::Item(_));
                }
            }
        }
        maybe_lint_redundant_semis(cx, &mut seq, after_item_stmt);
    }
}

fn maybe_lint_redundant_semis(
    cx: &EarlyContext<'_>,
    seq: &mut Option<(Span, bool)>,
    after_item_stmt: bool,
) {
    if let Some((span, multiple)) = seq.take() {
        // FIXME: Find a better way of ignoring the trailing
        // semicolon from macro expansion
        if span == rustc_span::DUMMY_SP {
            return;
        }

        // FIXME: Lint on semicolons after item statements
        // once doing so doesn't break bootstrapping
        if after_item_stmt {
            return;
        }

        cx.struct_span_lint(REDUNDANT_SEMICOLONS, span, |lint| {
            let (msg, rem) = if multiple {
                ("unnecessary trailing semicolons", "remove these semicolons")
            } else {
                ("unnecessary trailing semicolon", "remove this semicolon")
            };
            lint.build(msg)
                .span_suggestion(span, rem, String::new(), Applicability::MaybeIncorrect)
                .emit();
        });
    }
}
