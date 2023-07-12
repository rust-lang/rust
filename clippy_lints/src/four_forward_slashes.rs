use clippy_utils::{diagnostics::span_lint_and_sugg, source::snippet_opt};
use rustc_errors::Applicability;
use rustc_hir::Item;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{Span, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for outer doc comments written with 4 forward slashes (`////`).
    ///
    /// ### Why is this bad?
    /// This is (probably) a typo, and results in it not being a doc comment; just a regular
    /// comment.
    ///
    /// ### Example
    /// ```rust
    /// //// My amazing data structure
    /// pub struct Foo {
    ///     // ...
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// /// My amazing data structure
    /// pub struct Foo {
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "1.72.0"]
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
        let src = cx.sess().source_map();
        let item_and_attrs_span = cx
            .tcx
            .hir()
            .attrs(item.hir_id())
            .iter()
            .fold(item.span.shrink_to_lo(), |span, attr| span.to(attr.span));
        let (Some(file), _, _, end_line, _) = src.span_to_location_info(item_and_attrs_span) else {
            return;
        };
        for line in (0..end_line.saturating_sub(1)).rev() {
            let Some(contents) = file.get_line(line) else {
                continue;
            };
            let contents = contents.trim();
            if contents.is_empty() {
                break;
            }
            if contents.starts_with("////") {
                let bounds = file.line_bounds(line);
                let span = Span::new(bounds.start, bounds.end, SyntaxContext::root(), None);

                if snippet_opt(cx, span).is_some_and(|s| s.trim().starts_with("////")) {
                    span_lint_and_sugg(
                        cx,
                        FOUR_FORWARD_SLASHES,
                        span,
                        "comment with 4 forward slashes (`////`). This looks like a doc comment, but it isn't",
                        "make this a doc comment by removing one `/`",
                        // It's a little unfortunate but the span includes the `\n` yet the contents
                        // do not, so we must add it back. If some codebase uses `\r\n` instead they
                        // will need normalization but it should be fine
                        contents.replacen("////", "///", 1) + "\n",
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
    }
}
