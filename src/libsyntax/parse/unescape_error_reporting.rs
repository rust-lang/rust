//! Utilities for rendering escape sequence errors as diagnostics.

use std::ops::Range;
use std::iter::once;

use syntax_pos::{Span, BytePos};

use crate::errors::{Handler, Applicability};

use super::unescape::{EscapeError, Mode};

pub(crate) fn emit_unescape_error(
    handler: &Handler,
    // interior part of the literal, without quotes
    lit: &str,
    // full span of the literal, including quotes
    span_with_quotes: Span,
    mode: Mode,
    // range of the error inside `lit`
    range: Range<usize>,
    error: EscapeError,
) {
    log::debug!("emit_unescape_error: {:?}, {:?}, {:?}, {:?}, {:?}",
                lit, span_with_quotes, mode, range, error);
    let span = {
        let Range { start, end } = range;
        let (start, end) = (start as u32, end as u32);
        let lo = span_with_quotes.lo() + BytePos(start + 1);
        let hi = lo + BytePos(end - start);
            span_with_quotes
            .with_lo(lo)
            .with_hi(hi)
    };
    let last_char = || {
        let c = lit[range.clone()].chars().rev().next().unwrap();
        let span = span.with_lo(span.hi() - BytePos(c.len_utf8() as u32));
        (c, span)
    };
    match error {
        EscapeError::LoneSurrogateUnicodeEscape => {
            handler.struct_span_err(span, "invalid unicode character escape")
                .help("unicode escape must not be a surrogate")
                .emit();
        }
        EscapeError::OutOfRangeUnicodeEscape => {
            handler.struct_span_err(span, "invalid unicode character escape")
                .help("unicode escape must be at most 10FFFF")
                .emit();
        }
        EscapeError::MoreThanOneChar => {
            handler
                .struct_span_err(
                    span_with_quotes,
                    "character literal may only contain one codepoint",
                )
                .span_suggestion(
                    span_with_quotes,
                    "if you meant to write a `str` literal, use double quotes",
                    format!("\"{}\"", lit),
                    Applicability::MachineApplicable,
                ).emit()
        }
        EscapeError::EscapeOnlyChar => {
            let (c, _span) = last_char();

            let mut msg = if mode.is_bytes() {
                "byte constant must be escaped: "
            } else {
                "character constant must be escaped: "
            }.to_string();
            push_escaped_char(&mut msg, c);

            handler.span_err(span, msg.as_str())
        }
        EscapeError::BareCarriageReturn => {
            let msg = if mode.in_double_quotes() {
                "bare CR not allowed in string, use \\r instead"
            } else {
                "character constant must be escaped: \\r"
            };
            handler.span_err(span, msg);
        }
        EscapeError::BareCarriageReturnInRawString => {
            assert!(mode.in_double_quotes());
            let msg = "bare CR not allowed in raw string";
            handler.span_err(span, msg);
        }
        EscapeError::InvalidEscape => {
            let (c, span) = last_char();

            let label = if mode.is_bytes() {
                "unknown byte escape"
            } else {
                "unknown character escape"
            };
            let mut msg = label.to_string();
            msg.push_str(": ");
            push_escaped_char(&mut msg, c);

            let mut diag = handler.struct_span_err(span, msg.as_str());
            diag.span_label(span, label);
            if c == '{' || c == '}' && !mode.is_bytes() {
                diag.help("if used in a formatting string, \
                           curly braces are escaped with `{{` and `}}`");
            } else if c == '\r' {
                diag.help("this is an isolated carriage return; \
                           consider checking your editor and version control settings");
            }
            diag.emit();
        }
        EscapeError::TooShortHexEscape => {
            handler.span_err(span, "numeric character escape is too short")
        }
        EscapeError::InvalidCharInHexEscape | EscapeError::InvalidCharInUnicodeEscape => {
            let (c, span) = last_char();

            let mut msg = if error == EscapeError::InvalidCharInHexEscape {
                "invalid character in numeric character escape: "
            } else {
                "invalid character in unicode escape: "
            }.to_string();
            push_escaped_char(&mut msg, c);

            handler.span_err(span, msg.as_str())
        }
        EscapeError::NonAsciiCharInByte => {
            assert!(mode.is_bytes());
            let (_c, span) = last_char();
            handler.span_err(span, "byte constant must be ASCII. \
                                    Use a \\xHH escape for a non-ASCII byte")
        }
        EscapeError::NonAsciiCharInByteString => {
            assert!(mode.is_bytes());
            let (_c, span) = last_char();
            handler.span_err(span, "raw byte string must be ASCII")
        }
        EscapeError::OutOfRangeHexEscape => {
            handler.span_err(span, "this form of character escape may only be used \
                                    with characters in the range [\\x00-\\x7f]")
        }
        EscapeError::LeadingUnderscoreUnicodeEscape => {
            let (_c, span) = last_char();
            handler.span_err(span, "invalid start of unicode escape")
        }
        EscapeError::OverlongUnicodeEscape => {
            handler.span_err(span, "overlong unicode escape (must have at most 6 hex digits)")
        }
        EscapeError::UnclosedUnicodeEscape => {
            handler.span_err(span, "unterminated unicode escape (needed a `}`)")
        }
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
                let lo = char_span.lo();
                let hi = lo + BytePos(suggestion_len as u32);
                diag.span_suggestion(
                    span.with_lo(lo).with_hi(hi),
                    "format of unicode escape sequences uses braces",
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            } else {
                diag.span_label(span, msg);
                diag.help(
                    "format of unicode escape sequences is `\\u{...}`",
                );
            }

            diag.emit();
        }
        EscapeError::UnicodeEscapeInByte => {
            handler.span_err(span, "unicode escape sequences cannot be used \
                                    as a byte or in a byte string")
        }
        EscapeError::EmptyUnicodeEscape => {
            handler.span_err(span, "empty unicode escape (must have at least 1 hex digit)")
        }
        EscapeError::ZeroChars => {
            handler.span_err(span, "empty character literal")
        }
        EscapeError::LoneSlash => {
            panic!("lexer accepted unterminated literal with trailing slash")
        }
    }
}

/// Pushes a character to a message string for error reporting
pub(crate) fn push_escaped_char(msg: &mut String, c: char) {
    match c {
        '\u{20}'..='\u{7e}' => {
            // Don't escape \, ' or " for user-facing messages
            msg.push(c);
        }
        _ => {
            msg.extend(c.escape_default());
        }
    }
}
