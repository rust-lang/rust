use crate::utils::span_lint_and_sugg;
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;
use syntax::ast;

/// **What it does:** Checks for usage of dbg!() macro.
///
/// **Why is this bad?** `dbg!` macro is intended as a debugging tool. It
/// should not be in version control.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust,ignore
/// // Bad
/// dbg!(true)
///
/// // Good
/// true
/// ```
declare_clippy_lint! {
    pub DBG_MACRO,
    restriction,
    "`dbg!` macro is intended as a debugging tool"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DBG_MACRO)
    }

    fn name(&self) -> &'static str {
        "DbgMacro"
    }
}

impl EarlyLintPass for Pass {
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &ast::Mac) {
        if mac.node.path == "dbg" {
            span_lint_and_sugg(
                cx,
                DBG_MACRO,
                mac.span,
                "`dbg!` macro is intended as a debugging tool",
                "ensure to avoid having uses of it in version control",
                mac.node.tts.to_string(),
                Applicability::MaybeIncorrect,
            );
        }
    }
}
