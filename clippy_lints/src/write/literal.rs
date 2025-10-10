use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::format_arg_removal_span;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::sym;
use rustc_ast::token::LitKind;
use rustc_ast::{
    FormatArgPosition, FormatArgPositionKind, FormatArgs, FormatArgsPiece, FormatCount, FormatOptions,
    FormatPlaceholder, FormatTrait,
};
use rustc_errors::Applicability;
use rustc_lint::LateContext;
use rustc_span::Span;

use super::{PRINT_LITERAL, WRITE_LITERAL};

pub(super) fn check(cx: &LateContext<'_>, format_args: &FormatArgs, name: &str) {
    let arg_index = |argument: &FormatArgPosition| argument.index.unwrap_or_else(|pos| pos);

    let lint_name = if name.starts_with("write") {
        WRITE_LITERAL
    } else {
        PRINT_LITERAL
    };

    let mut counts = vec![0u32; format_args.arguments.all_args().len()];
    for piece in &format_args.template {
        if let FormatArgsPiece::Placeholder(placeholder) = piece {
            counts[arg_index(&placeholder.argument)] += 1;
        }
    }

    let mut suggestion: Vec<(Span, String)> = vec![];
    // holds index of replaced positional arguments; used to decrement the index of the remaining
    // positional arguments.
    let mut replaced_position: Vec<usize> = vec![];
    let mut sug_span: Option<Span> = None;

    for piece in &format_args.template {
        if let FormatArgsPiece::Placeholder(FormatPlaceholder {
            argument,
            span: Some(placeholder_span),
            format_trait: FormatTrait::Display,
            format_options,
        }) = piece
            && *format_options == FormatOptions::default()
            && let index = arg_index(argument)
            && counts[index] == 1
            && let Some(arg) = format_args.arguments.by_index(index)
            && let rustc_ast::ExprKind::Lit(lit) = &arg.expr.kind
            && !arg.expr.span.from_expansion()
            && let Some(value_string) = arg.expr.span.get_source_text(cx)
        {
            let (replacement, replace_raw) = match lit.kind {
                LitKind::Str | LitKind::StrRaw(_) => match extract_str_literal(&value_string) {
                    Some(extracted) => extracted,
                    None => return,
                },
                LitKind::Char => (
                    match lit.symbol {
                        sym::DOUBLE_QUOTE => "\\\"",
                        sym::BACKSLASH_SINGLE_QUOTE => "'",
                        _ => match value_string.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
                            Some(stripped) => stripped,
                            None => return,
                        },
                    }
                    .to_string(),
                    false,
                ),
                LitKind::Bool => (lit.symbol.to_string(), false),
                _ => continue,
            };

            let Some(format_string_snippet) = format_args.span.get_source_text(cx) else {
                continue;
            };
            let format_string_is_raw = format_string_snippet.starts_with('r');

            let replacement = match (format_string_is_raw, replace_raw) {
                (false, false) => Some(replacement),
                (false, true) => Some(replacement.replace('\\', "\\\\").replace('"', "\\\"")),
                (true, false) => match conservative_unescape(&replacement) {
                    Ok(unescaped) => Some(unescaped),
                    Err(UnescapeErr::Lint) => None,
                    Err(UnescapeErr::Ignore) => continue,
                },
                (true, true) => {
                    if replacement.contains(['#', '"']) {
                        None
                    } else {
                        Some(replacement)
                    }
                },
            };

            sug_span = Some(sug_span.unwrap_or(arg.expr.span).to(arg.expr.span));

            if let Some((_, index)) = format_arg_piece_span(piece) {
                replaced_position.push(index);
            }

            if let Some(replacement) = replacement
                // `format!("{}", "a")`, `format!("{named}", named = "b")
                //              ~~~~~                      ~~~~~~~~~~~~~
                && let Some(removal_span) = format_arg_removal_span(format_args, index)
            {
                let replacement = escape_braces(&replacement, !format_string_is_raw && !replace_raw);
                suggestion.push((*placeholder_span, replacement));
                suggestion.push((removal_span, String::new()));
            }
        }
    }

    // Decrement the index of the remaining by the number of replaced positional arguments
    if !suggestion.is_empty() {
        for piece in &format_args.template {
            relocalize_format_args_indexes(piece, &mut suggestion, &replaced_position);
        }
    }

    if let Some(span) = sug_span {
        span_lint_and_then(cx, lint_name, span, "literal with an empty format string", |diag| {
            if !suggestion.is_empty() {
                diag.multipart_suggestion("try", suggestion, Applicability::MachineApplicable);
            }
        });
    }
}

/// Extract Span and its index from the given `piece`
fn format_arg_piece_span(piece: &FormatArgsPiece) -> Option<(Span, usize)> {
    match piece {
        FormatArgsPiece::Placeholder(FormatPlaceholder {
            argument: FormatArgPosition { index: Ok(index), .. },
            span: Some(span),
            ..
        }) => Some((*span, *index)),
        _ => None,
    }
}

/// Relocalizes the indexes of positional arguments in the format string
fn relocalize_format_args_indexes(
    piece: &FormatArgsPiece,
    suggestion: &mut Vec<(Span, String)>,
    replaced_position: &[usize],
) {
    if let FormatArgsPiece::Placeholder(FormatPlaceholder {
        argument:
            FormatArgPosition {
                index: Ok(index),
                // Only consider positional arguments
                kind: FormatArgPositionKind::Number,
                span: Some(span),
            },
        format_options,
        ..
    }) = piece
    {
        if suggestion.iter().any(|(s, _)| s.overlaps(*span)) {
            // If the span is already in the suggestion, we don't need to process it again
            return;
        }

        // lambda to get the decremented index based on the replaced positions
        let decremented_index = |index: usize| -> usize {
            let decrement = replaced_position.iter().filter(|&&i| i < index).count();
            index - decrement
        };

        suggestion.push((*span, decremented_index(*index).to_string()));

        // If there are format options, we need to handle them as well
        if *format_options != FormatOptions::default() {
            // lambda to process width and precision format counts and add them to the suggestion
            let mut process_format_count = |count: &Option<FormatCount>, formatter: &dyn Fn(usize) -> String| {
                if let Some(FormatCount::Argument(FormatArgPosition {
                    index: Ok(format_arg_index),
                    kind: FormatArgPositionKind::Number,
                    span: Some(format_arg_span),
                })) = count
                {
                    suggestion.push((*format_arg_span, formatter(decremented_index(*format_arg_index))));
                }
            };

            process_format_count(&format_options.width, &|index: usize| format!("{index}$"));
            process_format_count(&format_options.precision, &|index: usize| format!(".{index}$"));
        }
    }
}

/// Removes the raw marker, `#`s and quotes from a str, and returns if the literal is raw
///
/// `r#"a"#` -> (`a`, true)
///
/// `"b"` -> (`b`, false)
fn extract_str_literal(literal: &str) -> Option<(String, bool)> {
    let (literal, raw) = match literal.strip_prefix('r') {
        Some(stripped) => (stripped.trim_matches('#'), true),
        None => (literal, false),
    };

    Some((literal.strip_prefix('"')?.strip_suffix('"')?.to_string(), raw))
}

enum UnescapeErr {
    /// Should still be linted, can be manually resolved by author, e.g.
    ///
    /// ```ignore
    /// print!(r"{}", '"');
    /// ```
    Lint,
    /// Should not be linted, e.g.
    ///
    /// ```ignore
    /// print!(r"{}", '\r');
    /// ```
    Ignore,
}

/// Unescape a normal string into a raw string
fn conservative_unescape(literal: &str) -> Result<String, UnescapeErr> {
    let mut unescaped = String::with_capacity(literal.len());
    let mut chars = literal.chars();
    let mut err = false;

    while let Some(ch) = chars.next() {
        match ch {
            '#' => err = true,
            '\\' => match chars.next() {
                Some('\\') => unescaped.push('\\'),
                Some('"') => err = true,
                _ => return Err(UnescapeErr::Ignore),
            },
            _ => unescaped.push(ch),
        }
    }

    if err { Err(UnescapeErr::Lint) } else { Ok(unescaped) }
}

/// Replaces `{` with `{{` and `}` with `}}`. If `preserve_unicode_escapes` is `true` the braces
/// in `\u{xxxx}` are left unmodified
#[expect(clippy::match_same_arms)]
fn escape_braces(literal: &str, preserve_unicode_escapes: bool) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        Backslash,
        UnicodeEscape,
    }

    let mut escaped = String::with_capacity(literal.len());
    let mut state = State::Normal;

    for ch in literal.chars() {
        state = match (ch, state) {
            // Escape braces outside of unicode escapes by doubling them up
            ('{' | '}', State::Normal) => {
                escaped.push(ch);
                State::Normal
            },
            // If `preserve_unicode_escapes` isn't enabled stay in `State::Normal`, otherwise:
            //
            // \u{aaaa} \\ \x01
            // ^        ^  ^
            ('\\', State::Normal) if preserve_unicode_escapes => State::Backslash,
            // \u{aaaa}
            //  ^
            ('u', State::Backslash) => State::UnicodeEscape,
            // \xAA \\
            //  ^    ^
            (_, State::Backslash) => State::Normal,
            // \u{aaaa}
            //        ^
            ('}', State::UnicodeEscape) => State::Normal,
            _ => state,
        };

        escaped.push(ch);
    }

    escaped
}
