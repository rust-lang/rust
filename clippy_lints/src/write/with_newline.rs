use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::MacroCall;
use clippy_utils::source::{SpanRangeExt, expand_past_previous_comma};
use clippy_utils::sym;
use rustc_ast::{FormatArgs, FormatArgsPiece};
use rustc_errors::Applicability;
use rustc_lint::{LateContext, LintContext};
use rustc_span::BytePos;

use super::{PRINT_WITH_NEWLINE, WRITE_WITH_NEWLINE};

pub(super) fn check(cx: &LateContext<'_>, format_args: &FormatArgs, macro_call: &MacroCall, name: &str) {
    let Some(&FormatArgsPiece::Literal(last)) = format_args.template.last() else {
        return;
    };

    let count_vertical_whitespace = || {
        format_args
            .template
            .iter()
            .filter_map(|piece| match piece {
                FormatArgsPiece::Literal(literal) => Some(literal),
                FormatArgsPiece::Placeholder(_) => None,
            })
            .flat_map(|literal| literal.as_str().chars())
            .filter(|ch| matches!(ch, '\r' | '\n'))
            .count()
    };

    if last.as_str().ends_with('\n')
        // ignore format strings with other internal vertical whitespace
        && count_vertical_whitespace() == 1
    {
        let mut format_string_span = format_args.span;

        let lint = if name == "write" {
            format_string_span = expand_past_previous_comma(cx, format_string_span);

            WRITE_WITH_NEWLINE
        } else {
            PRINT_WITH_NEWLINE
        };

        span_lint_and_then(
            cx,
            lint,
            macro_call.span,
            format!("using `{name}!()` with a format string that ends in a single newline"),
            |diag| {
                let name_span = cx.sess().source_map().span_until_char(macro_call.span, '!');
                let Some(format_snippet) = format_string_span.get_source_text(cx) else {
                    return;
                };

                if format_args.template.len() == 1 && last == sym::LF {
                    // print!("\n"), write!(f, "\n")

                    diag.multipart_suggestion(
                        format!("use `{name}ln!` instead"),
                        vec![(name_span, format!("{name}ln")), (format_string_span, String::new())],
                        Applicability::MachineApplicable,
                    );
                } else if format_snippet.ends_with("\\n\"") {
                    // print!("...\n"), write!(f, "...\n")

                    let hi = format_string_span.hi();
                    let newline_span = format_string_span.with_lo(hi - BytePos(3)).with_hi(hi - BytePos(1));

                    diag.multipart_suggestion(
                        format!("use `{name}ln!` instead"),
                        vec![(name_span, format!("{name}ln")), (newline_span, String::new())],
                        Applicability::MachineApplicable,
                    );
                }
            },
        );
    }
}
