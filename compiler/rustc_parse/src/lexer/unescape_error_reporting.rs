//! Utilities for rendering escape sequence errors as diagnostics.

use std::iter::once;
use std::ops::Range;

use rustc_errors::{pluralize, Applicability, Handler};
use rustc_lexer::unescape::{EscapeError, Mode};
use rustc_span::{BytePos, Span};

pub(crate) fn emit_unescape_error(
    handler: &Handler,
    // interior part of the literal, without quotes
    lit: &str,
    // full span of the literal, including quotes
    span_with_quotes: Span,
    // interior span of the literal, without quotes
    span: Span,
    mode: Mode,
    // range of the error inside `lit`
    range: Range<usize>,
    error: EscapeError,
) {
    tracing::debug!(
        "emit_unescape_error: {:?}, {:?}, {:?}, {:?}, {:?}",
        lit,
        span_with_quotes,
        mode,
        range,
        error
    );
    let last_char = || {
        let c = lit[range.clone()].chars().rev().next().unwrap();
        let span = span.with_lo(span.hi() - BytePos(c.len_utf8() as u32));
        (c, span)
    };
    match error {
        EscapeError::LoneSurrogateUnicodeEscape => {
            handler
                .struct_span_err(span, "invalid unicode character escape")
                .span_label(span, "invalid escape")
                .help("unicode escape must not be a surrogate")
                .emit();
        }
        EscapeError::OutOfRangeUnicodeEscape => {
            handler
                .struct_span_err(span, "invalid unicode character escape")
                .span_label(span, "invalid escape")
                .help("unicode escape must be at most 10FFFF")
                .emit();
        }
        EscapeError::MoreThanOneChar => {
            use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

            let mut has_help = false;
            let mut handler = handler.struct_span_err(
                span_with_quotes,
                "character literal may only contain one codepoint",
            );

            if lit.chars().skip(1).all(|c| is_combining_mark(c)) {
                let escaped_marks =
                    lit.chars().skip(1).map(|c| c.escape_default().to_string()).collect::<Vec<_>>();
                handler.span_note(
                    span,
                    &format!(
                        "this `{}` is followed by the combining mark{} `{}`",
                        lit.chars().next().unwrap(),
                        pluralize!(escaped_marks.len()),
                        escaped_marks.join(""),
                    ),
                );
                let normalized = lit.nfc().to_string();
                if normalized.chars().count() == 1 {
                    has_help = true;
                    handler.span_suggestion(
                        span,
                        &format!(
                            "consider using the normalized form `{}` of this character",
                            normalized.chars().next().unwrap().escape_default()
                        ),
                        normalized,
                        Applicability::MachineApplicable,
                    );
                }
            }

            if !has_help {
                let (prefix, msg) = if mode.is_bytes() {
                    ("b", "if you meant to write a byte string literal, use double quotes")
                } else {
                    ("", "if you meant to write a `str` literal, use double quotes")
                };

                handler.span_suggestion(
                    span_with_quotes,
                    msg,
                    format!("{}\"{}\"", prefix, lit),
                    Applicability::MachineApplicable,
                );
            }

            handler.emit();
        }
        EscapeError::EscapeOnlyChar => {
            let (c, char_span) = last_char();

            let msg = if mode.is_bytes() {
                "byte constant must be escaped"
            } else {
                "character constant must be escaped"
            };
            handler
                .struct_span_err(span, &format!("{}: `{}`", msg, escaped_char(c)))
                .span_suggestion(
                    char_span,
                    "escape the character",
                    c.escape_default().to_string(),
                    Applicability::MachineApplicable,
                )
                .emit()
        }
        EscapeError::BareCarriageReturn => {
            let msg = if mode.in_double_quotes() {
                "bare CR not allowed in string, use `\\r` instead"
            } else {
                "character constant must be escaped: `\\r`"
            };
            handler
                .struct_span_err(span, msg)
                .span_suggestion(
                    span,
                    "escape the character",
                    "\\r".to_string(),
                    Applicability::MachineApplicable,
                )
                .emit();
        }
        EscapeError::BareCarriageReturnInRawString => {
            assert!(mode.in_double_quotes());
            let msg = "bare CR not allowed in raw string";
            handler.span_err(span, msg);
        }
        EscapeError::InvalidEscape => {
            let (c, span) = last_char();

            let label =
                if mode.is_bytes() { "unknown byte escape" } else { "unknown character escape" };
            let ec = escaped_char(c);
            let mut diag = handler.struct_span_err(span, &format!("{}: `{}`", label, ec));
            diag.span_label(span, label);
            if c == '{' || c == '}' && !mode.is_bytes() {
                diag.help(
                    "if used in a formatting string, curly braces are escaped with `{{` and `}}`",
                );
            } else if c == '\r' {
                diag.help(
                    "this is an isolated carriage return; consider checking your editor and \
                     version control settings",
                );
            } else {
                diag.help(
                    "for more information, visit \
                     <https://static.rust-lang.org/doc/master/reference.html#literals>",
                );
            }
            diag.emit();
        }
        EscapeError::TooShortHexEscape => {
            handler.span_err(span, "numeric character escape is too short")
        }
        EscapeError::InvalidCharInHexEscape | EscapeError::InvalidCharInUnicodeEscape => {
            let (c, span) = last_char();

            let msg = if error == EscapeError::InvalidCharInHexEscape {
                "invalid character in numeric character escape"
            } else {
                "invalid character in unicode escape"
            };
            let c = escaped_char(c);

            handler
                .struct_span_err(span, &format!("{}: `{}`", msg, c))
                .span_label(span, msg)
                .emit();
        }
        EscapeError::NonAsciiCharInByte => {
            assert!(mode.is_bytes());
            let (c, span) = last_char();
            let mut err = handler.struct_span_err(span, "non-ASCII character in byte constant");
            err.span_label(span, "byte constant must be ASCII");
            if (c as u32) <= 0xFF {
                err.span_suggestion(
                    span,
                    &format!(
                        "if you meant to use the unicode code point for '{}', use a \\xHH escape",
                        c
                    ),
                    format!("\\x{:X}", c as u32),
                    Applicability::MaybeIncorrect,
                );
            } else if matches!(mode, Mode::Byte) {
                err.span_label(span, "this multibyte character does not fit into a single byte");
            } else if matches!(mode, Mode::ByteStr) {
                let mut utf8 = String::new();
                utf8.push(c);
                err.span_suggestion(
                    span,
                    &format!(
                        "if you meant to use the UTF-8 encoding of '{}', use \\xHH escapes",
                        c
                    ),
                    utf8.as_bytes()
                        .iter()
                        .map(|b: &u8| format!("\\x{:X}", *b))
                        .fold("".to_string(), |a, c| a + &c),
                    Applicability::MaybeIncorrect,
                );
            }
            err.emit();
        }
        EscapeError::NonAsciiCharInByteString => {
            assert!(mode.is_bytes());
            let (_c, span) = last_char();
            handler
                .struct_span_err(span, "raw byte string must be ASCII")
                .span_label(span, "must be ASCII")
                .emit();
        }
        EscapeError::OutOfRangeHexEscape => {
            handler
                .struct_span_err(span, "out of range hex escape")
                .span_label(span, "must be a character in the range [\\x00-\\x7f]")
                .emit();
        }
        EscapeError::LeadingUnderscoreUnicodeEscape => {
            let (c, span) = last_char();
            let msg = "invalid start of unicode escape";
            handler
                .struct_span_err(span, &format!("{}: `{}`", msg, c))
                .span_label(span, msg)
                .emit();
        }
        EscapeError::OverlongUnicodeEscape => {
            handler
                .struct_span_err(span, "overlong unicode escape")
                .span_label(span, "must have at most 6 hex digits")
                .emit();
        }
        EscapeError::UnclosedUnicodeEscape => handler
            .struct_span_err(span, "unterminated unicode escape")
            .span_label(span, "missing a closing `}`")
            .span_suggestion_verbose(
                span.shrink_to_hi(),
                "terminate the unicode escape",
                "}".to_string(),
                Applicability::MaybeIncorrect,
            )
            .emit(),
        EscapeError::NoBraceInUnicodeEscape => {
            let msg = "incorrect unicode escape sequence";
            let mut diag = handler.struct_span_err(span, msg);

            let mut suggestion = "\\u{".to_owned();
            let mut suggestion_len = 0;
            let (c, char_span) = last_char();
            let chars = once(c).chain(lit[range.end..].chars());
            for c in chars.take(6).take_while(|c| c.is_digit(16)) {
                suggestion.push(c);
                suggestion_len += c.len_utf8();
            }

            if suggestion_len > 0 {
                suggestion.push('}');
                let hi = char_span.lo() + BytePos(suggestion_len as u32);
                diag.span_suggestion(
                    span.with_hi(hi),
                    "format of unicode escape sequences uses braces",
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            } else {
                diag.span_label(span, msg);
                diag.help("format of unicode escape sequences is `\\u{...}`");
            }

            diag.emit();
        }
        EscapeError::UnicodeEscapeInByte => {
            let msg = "unicode escape in byte string";
            handler
                .struct_span_err(span, msg)
                .span_label(span, msg)
                .help("unicode escape sequences cannot be used as a byte or in a byte string")
                .emit();
        }
        EscapeError::EmptyUnicodeEscape => {
            handler
                .struct_span_err(span, "empty unicode escape")
                .span_label(span, "this escape must have at least 1 hex digit")
                .emit();
        }
        EscapeError::ZeroChars => {
            let msg = "empty character literal";
            handler.struct_span_err(span, msg).span_label(span, msg).emit()
        }
        EscapeError::LoneSlash => {
            let msg = "invalid trailing slash in literal";
            handler.struct_span_err(span, msg).span_label(span, msg).emit();
        }
        EscapeError::UnskippedWhitespaceWarning => {
            let (c, char_span) = last_char();
            let msg =
                format!("non-ASCII whitespace symbol '{}' is not skipped", c.escape_unicode());
            handler.struct_span_warn(span, &msg).span_label(char_span, &msg).emit();
        }
        EscapeError::MultipleSkippedLinesWarning => {
            let msg = "multiple lines skipped by escaped newline";
            let bottom_msg = "skipping everything up to and including this point";
            handler.struct_span_warn(span, msg).span_label(span, bottom_msg).emit();
        }
    }
}

/// Pushes a character to a message string for error reporting
pub(crate) fn escaped_char(c: char) -> String {
    match c {
        '\u{20}'..='\u{7e}' => {
            // Don't escape \, ' or " for user-facing messages
            c.to_string()
        }
        _ => c.escape_default().to_string(),
    }
}
