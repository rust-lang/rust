use rustc_ast::{Block, StmtKind};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;

use crate::lints::{RedundantSemicolonsDiag, RedundantSemicolonsSuggestion};
use crate::{EarlyContext, EarlyLintPass, LintContext};

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
        let mut seq = None;
        for stmt in block.stmts.iter() {
            match (&stmt.kind, &mut seq) {
                (StmtKind::Empty, None) => seq = Some((stmt.span, false)),
                (StmtKind::Empty, Some(seq)) => *seq = (seq.0.to(stmt.span), true),
                (_, seq) => maybe_lint_redundant_semis(cx, seq),
            }
        }
        maybe_lint_redundant_semis(cx, &mut seq);
    }
}

fn maybe_lint_redundant_semis(cx: &EarlyContext<'_>, seq: &mut Option<(Span, bool)>) {
    if let Some((span, multiple)) = seq.take() {
        if span == rustc_span::DUMMY_SP {
            return;
        }

        // Ignore redundant semicolons inside macro expansion.(issue #142143)
        let suggestion = if span.from_expansion() {
            None
        } else {
            Some(RedundantSemicolonsSuggestion { multiple_semicolons: multiple, span })
        };

        cx.emit_span_lint(
            REDUNDANT_SEMICOLONS,
            span,
            RedundantSemicolonsDiag { multiple, suggestion },
        );
    }
}
