use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanRangeExt as _;
use itertools::Itertools;
use rustc_errors::Applicability;
use rustc_hir::Item;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for outer doc comments written with 4 forward slashes (`////`).
    ///
    /// ### Why is this bad?
    /// This is (probably) a typo, and results in it not being a doc comment; just a regular
    /// comment.
    ///
    /// ### Example
    /// ```no_run
    /// //// My amazing data structure
    /// pub struct Foo {
    ///     // ...
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// /// My amazing data structure
    /// pub struct Foo {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "1.73.0"]
    pub FOUR_FORWARD_SLASHES,
    suspicious,
    "comments with 4 forward slashes (`////`) likely intended to be doc comments (`///`)"
}
declare_lint_pass!(FourForwardSlashes => [FOUR_FORWARD_SLASHES]);

impl<'tcx> LateLintPass<'tcx> for FourForwardSlashes {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if item.span.from_expansion() {
            return;
        }
        let sm = cx.sess().source_map();
        let mut span = cx
            .tcx
            .hir_attrs(item.hir_id())
            .iter()
            .filter(|i| i.is_doc_comment())
            .fold(item.span.shrink_to_lo(), |span, attr| span.to(attr.span()));
        let (Some(file), _, _, end_line, _) = sm.span_to_location_info(span) else {
            return;
        };
        let mut bad_comments = vec![];
        for line in (0..end_line.saturating_sub(1)).rev() {
            let Some(contents) = file.get_line(line).map(|c| c.trim().to_owned()) else {
                return;
            };
            // Keep searching until we find the next item
            if !contents.is_empty() && !contents.starts_with("//") && !contents.starts_with("#[") {
                break;
            }

            if contents.starts_with("////") && !matches!(contents.chars().nth(4), Some('/' | '!')) {
                let bounds = file.line_bounds(line);
                let line_span = Span::with_root_ctxt(bounds.start, bounds.end);
                span = line_span.to(span);
                bad_comments.push((line_span, contents));
            }
        }

        if !bad_comments.is_empty() {
            span_lint_and_then(
                cx,
                FOUR_FORWARD_SLASHES,
                span,
                "this item has comments with 4 forward slashes (`////`). These look like doc comments, but they aren't",
                |diag| {
                    let msg = if bad_comments.len() == 1 {
                        "make this a doc comment by removing one `/`"
                    } else {
                        "turn these into doc comments by removing one `/`"
                    };

                    // If the comment contains a bare CR (not followed by a LF), do not propose an auto-fix
                    // as bare CR are not allowed in doc comments.
                    if span.check_source_text(cx, contains_bare_cr) {
                        diag.help(msg)
                            .note("bare CR characters are not allowed in doc comments");
                        return;
                    }

                    diag.multipart_suggestion(
                        msg,
                        bad_comments
                            .into_iter()
                            // It's a little unfortunate but the span includes the `\n` yet the contents
                            // do not, so we must add it back. If some codebase uses `\r\n` instead they
                            // will need normalization but it should be fine
                            .map(|(span, c)| (span, c.replacen("////", "///", 1) + "\n"))
                            .collect(),
                        Applicability::MachineApplicable,
                    );
                },
            );
        }
    }
}

/// Checks if `text` contains any CR not followed by a LF
fn contains_bare_cr(text: &str) -> bool {
    text.bytes().tuple_windows().any(|(a, b)| a == b'\r' && b != b'\n')
}
