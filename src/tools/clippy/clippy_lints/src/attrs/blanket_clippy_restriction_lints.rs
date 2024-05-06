use super::utils::extract_clippy_lint;
use super::BLANKET_CLIPPY_RESTRICTION_LINTS;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use rustc_ast::NestedMetaItem;
use rustc_lint::{LateContext, Level, LintContext};
use rustc_span::symbol::Symbol;
use rustc_span::{sym, DUMMY_SP};

pub(super) fn check(cx: &LateContext<'_>, name: Symbol, items: &[NestedMetaItem]) {
    for lint in items {
        if let Some(lint_name) = extract_clippy_lint(lint) {
            if lint_name.as_str() == "restriction" && name != sym::allow {
                span_lint_and_help(
                    cx,
                    BLANKET_CLIPPY_RESTRICTION_LINTS,
                    lint.span(),
                    "`clippy::restriction` is not meant to be enabled as a group",
                    None,
                    "enable the restriction lints you need individually",
                );
            }
        }
    }
}

pub(super) fn check_command_line(cx: &LateContext<'_>) {
    for (name, level) in &cx.sess().opts.lint_opts {
        if name == "clippy::restriction" && *level > Level::Allow {
            span_lint_and_then(
                cx,
                BLANKET_CLIPPY_RESTRICTION_LINTS,
                DUMMY_SP,
                "`clippy::restriction` is not meant to be enabled as a group",
                |diag| {
                    diag.note(format!(
                        "because of the command line `--{} clippy::restriction`",
                        level.as_str()
                    ));
                    diag.help("enable the restriction lints you need individually");
                },
            );
        }
    }
}
