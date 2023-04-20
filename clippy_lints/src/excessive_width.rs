use clippy_utils::diagnostics::span_lint_and_help;
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Pos;

// TODO: This still needs to be implemented.
declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for lines which are indented beyond a certain threshold.
    ///
    /// ### Why is this bad?
    ///
    /// It can severely hinder readability. The default is very generous; if you
    /// exceed this, it's a sign you should refactor.
    ///
    /// ### Example
    /// TODO
    /// Use instead:
    /// TODO
    #[clippy::version = "1.70.0"]
    pub EXCESSIVE_INDENTATION,
    style,
    "check for lines intended beyond a certain threshold"
}
declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for lines which are longer than a certain threshold.
    ///
    /// ### Why is this bad?
    ///
    /// It can severely hinder readability. Almost always, running rustfmt will get this
    /// below this threshold (or whatever you have set as max_width), but if it fails,
    /// it's probably a sign you should refactor.
    ///
    /// ### Example
    /// TODO
    /// Use instead:
    /// TODO
    #[clippy::version = "1.70.0"]
    pub EXCESSIVE_WIDTH,
    style,
    "check for lines longer than a certain threshold"
}
impl_lint_pass!(ExcessiveWidth => [EXCESSIVE_INDENTATION, EXCESSIVE_WIDTH]);

#[derive(Clone, Copy)]
pub struct ExcessiveWidth {
    pub excessive_width_threshold: u64,
    pub excessive_width_ignore_indentation: bool,
    pub excessive_indentation_threshold: u64,
}

impl LateLintPass<'_> for ExcessiveWidth {
    fn check_stmt(&mut self, cx: &LateContext<'_>, stmt: &Stmt<'_>) {
        if in_external_macro(cx.sess(), stmt.span) {
            return;
        }

        if let Ok(lines) = cx.sess().source_map().span_to_lines(stmt.span).map(|info| info.lines) {
            for line in &lines {
                // TODO: yeah, no.
                if (line.end_col.to_usize()
                    - line.start_col.to_usize() * self.excessive_width_ignore_indentation as usize)
                    > self.excessive_width_threshold as usize
                {
                    span_lint_and_help(
                        cx,
                        EXCESSIVE_WIDTH,
                        stmt.span,
                        "this line is too long",
                        None,
                        "consider running rustfmt or refactoring this",
                    );
                }
            }
        }
    }
}
