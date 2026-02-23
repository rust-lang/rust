use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::MacroCall;
use clippy_utils::source::expand_past_previous_comma;
use clippy_utils::{span_extract_comments, sym};
use rustc_ast::{FormatArgs, FormatArgsPiece};
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LintContext};

use super::{PRINTLN_EMPTY_STRING, WRITELN_EMPTY_STRING};

pub(super) fn check(cx: &LateContext<'_>, format_args: &FormatArgs, macro_call: &MacroCall, name: &str) {
    if let [FormatArgsPiece::Literal(sym::LF)] = &format_args.template[..] {
        let is_writeln = name == "writeln";

        span_lint_and_then(
            cx,
            if is_writeln {
                WRITELN_EMPTY_STRING
            } else {
                PRINTLN_EMPTY_STRING
            },
            macro_call.span,
            format!("empty string literal in `{name}!`"),
            |diag| {
                if span_extract_comments(cx.sess().source_map(), macro_call.span).is_empty() {
                    let closing_paren = cx.sess().source_map().span_extend_to_prev_char_before(
                        macro_call.span.shrink_to_hi(),
                        ')',
                        false,
                    );
                    let mut span = format_args.span.with_hi(closing_paren.lo());
                    if is_writeln {
                        span = expand_past_previous_comma(cx, span);
                    }

                    diag.span_suggestion(span, "remove the empty string", "", Applicability::MachineApplicable);
                } else {
                    // If there is a comment in the span of macro call, we don't provide an auto-fix suggestion.
                    diag.span_note(format_args.span, "remove the empty string");
                }
            },
        );
    }
}
