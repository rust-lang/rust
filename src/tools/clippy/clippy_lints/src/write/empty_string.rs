use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::MacroCall;
use clippy_utils::source::expand_past_previous_comma;
use clippy_utils::sym;
use rustc_ast::{FormatArgs, FormatArgsPiece};
use rustc_errors::Applicability;
use rustc_lint::LateContext;

use super::{PRINTLN_EMPTY_STRING, WRITELN_EMPTY_STRING};

pub(super) fn check(cx: &LateContext<'_>, format_args: &FormatArgs, macro_call: &MacroCall, name: &str) {
    if let [FormatArgsPiece::Literal(sym::LF)] = &format_args.template[..] {
        let mut span = format_args.span;

        let lint = if name == "writeln" {
            span = expand_past_previous_comma(cx, span);

            WRITELN_EMPTY_STRING
        } else {
            PRINTLN_EMPTY_STRING
        };

        span_lint_and_then(
            cx,
            lint,
            macro_call.span,
            format!("empty string literal in `{name}!`"),
            |diag| {
                diag.span_suggestion(
                    span,
                    "remove the empty string",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}
