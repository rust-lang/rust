//! Utilities for rendering escape sequence errors as diagnostics.

use std::iter::once;
use std::ops::Range;

use rustc_errors::{Applicability, DiagCtxtHandle, ErrorGuaranteed};
use rustc_literal_escaper::{EscapeError, Mode};
use rustc_span::{BytePos, Span};
use tracing::debug;

use crate::errors::{MoreThanOneCharNote, MoreThanOneCharSugg, NoBraceUnicodeSub, UnescapeError};

pub(crate) fn emit_unescape_error(
    dcx: DiagCtxtHandle<'_>,
    // interior part of the literal, between quotes
    lit: &str,
    // full span of the literal, including quotes and any prefix
    full_lit_span: Span,
    // span of the error part of the literal
    err_span: Span,
    mode: Mode,
    // range of the error inside `lit`
    range: Range<usize>,
    error: EscapeError,
) -> Option<ErrorGuaranteed> {
    debug!(
        "emit_unescape_error: {:?}, {:?}, {:?}, {:?}, {:?}",
        lit, full_lit_span, mode, range, error
    );
    let last_char = || {
        let c = lit[range.clone()].chars().next_back().unwrap();
        let span = err_span.with_lo(err_span.hi() - BytePos(c.len_utf8() as u32));
        (c, span)
    };
    Some(match error {
        EscapeError::LoneSurrogateUnicodeEscape => {
            dcx.emit_err(UnescapeError::InvalidUnicodeEscape { span: err_span, surrogate: true })
        }
        EscapeError::OutOfRangeUnicodeEscape => {
            dcx.emit_err(UnescapeError::InvalidUnicodeEscape { span: err_span, surrogate: false })
        }
        EscapeError::MoreThanOneChar => {
            use unicode_normalization::UnicodeNormalization;
            use unicode_normalization::char::is_combining_mark;
            let mut sugg = None;
            let mut note = None;

            let lit_chars = lit.chars().collect::<Vec<_>>();
            let (first, rest) = lit_chars.split_first().unwrap();
            if rest.iter().copied().all(is_combining_mark) {
                let normalized = lit.nfc().to_string();
                if normalized.chars().count() == 1 {
                    let ch = normalized.chars().next().unwrap().escape_default().to_string();
                    sugg = Some(MoreThanOneCharSugg::NormalizedForm {
                        span: err_span,
                        ch,
                        normalized,
                    });
                }
                let escaped_marks =
                    rest.iter().map(|c| c.escape_default().to_string()).collect::<Vec<_>>();
                note = Some(MoreThanOneCharNote::AllCombining {
                    span: err_span,
                    chr: format!("{first}"),
                    len: escaped_marks.len(),
                    escaped_marks: escaped_marks.join(""),
                });
            } else {
                let printable: Vec<char> = lit
                    .chars()
                    .filter(|&x| {
                        unicode_width::UnicodeWidthChar::width(x).unwrap_or(0) != 0
                            && !x.is_whitespace()
                    })
                    .collect();

                if let &[ch] = printable.as_slice() {
                    sugg = Some(MoreThanOneCharSugg::RemoveNonPrinting {
                        span: err_span,
                        ch: ch.to_string(),
                    });
                    note = Some(MoreThanOneCharNote::NonPrinting {
                        span: err_span,
                        escaped: lit.escape_default().to_string(),
                    });
                }
            };
            let sugg = sugg.unwrap_or_else(|| {
                let prefix = mode.prefix_noraw();
                let mut escaped = String::with_capacity(lit.len());
                let mut in_escape = false;
                for c in lit.chars() {
                    match c {
                        '\\' => in_escape = !in_escape,
                        '"' if !in_escape => escaped.push('\\'),
                        _ => in_escape = false,
                    }
                    escaped.push(c);
                }
                if escaped.len() != lit.len() || full_lit_span.is_empty() {
                    let sugg = format!("{prefix}\"{escaped}\"");
                    MoreThanOneCharSugg::QuotesFull {
                        span: full_lit_span,
                        is_byte: mode == Mode::Byte,
                        sugg,
                    }
                } else {
                    MoreThanOneCharSugg::Quotes {
                        start: full_lit_span
                            .with_hi(full_lit_span.lo() + BytePos((prefix.len() + 1) as u32)),
                        end: full_lit_span.with_lo(full_lit_span.hi() - BytePos(1)),
                        is_byte: mode == Mode::Byte,
                        prefix,
                    }
                }
            });
            dcx.emit_err(UnescapeError::MoreThanOneChar {
                span: full_lit_span,
                note,
                suggestion: sugg,
            })
        }
        EscapeError::EscapeOnlyChar => {
            let (c, char_span) = last_char();
            dcx.emit_err(UnescapeError::EscapeOnlyChar {
                span: err_span,
                char_span,
                escaped_sugg: c.escape_default().to_string(),
                escaped_msg: escaped_char(c),
                byte: mode == Mode::Byte,
            })
        }
        EscapeError::BareCarriageReturn => {
            let double_quotes = mode.in_double_quotes();
            dcx.emit_err(UnescapeError::BareCr { span: err_span, double_quotes })
        }
        EscapeError::BareCarriageReturnInRawString => {
            assert!(mode.in_double_quotes());
            dcx.emit_err(UnescapeError::BareCrRawString(err_span))
        }
        EscapeError::InvalidEscape => {
            let (c, span) = last_char();

            let label = if mode == Mode::Byte || mode == Mode::ByteStr {
                "unknown byte escape"
            } else {
                "unknown character escape"
            };
            let ec = escaped_char(c);
            let mut diag = dcx.struct_span_err(span, format!("{label}: `{ec}`"));
            diag.span_label(span, label);
            if c == '{' || c == '}' && matches!(mode, Mode::Str | Mode::RawStr) {
                diag.help(
                    "if used in a formatting string, curly braces are escaped with `{{` and `}}`",
                );
            } else if c == '\r' {
                diag.help(
                    "this is an isolated carriage return; consider checking your editor and \
                     version control settings",
                );
            } else {
                if mode == Mode::Str || mode == Mode::Char {
                    diag.span_suggestion(
                        full_lit_span,
                        "if you meant to write a literal backslash (perhaps escaping in a regular expression), consider a raw string literal",
                        format!("r\"{lit}\""),
                        Applicability::MaybeIncorrect,
                    );
                }

                diag.help(
                    "for more information, visit \
                     <https://doc.rust-lang.org/reference/tokens.html#literals>",
                );
            }
            diag.emit()
        }
        EscapeError::TooShortHexEscape => dcx.emit_err(UnescapeError::TooShortHexEscape(err_span)),
        EscapeError::InvalidCharInHexEscape | EscapeError::InvalidCharInUnicodeEscape => {
            let (c, span) = last_char();
            let is_hex = error == EscapeError::InvalidCharInHexEscape;
            let ch = escaped_char(c);
            dcx.emit_err(UnescapeError::InvalidCharInEscape { span, is_hex, ch })
        }
        EscapeError::NonAsciiCharInByte => {
            let (c, span) = last_char();
            let desc = match mode {
                Mode::Byte => "byte literal",
                Mode::ByteStr => "byte string literal",
                Mode::RawByteStr => "raw byte string literal",
                _ => panic!("non-is_byte literal paired with NonAsciiCharInByte"),
            };
            let mut err = dcx.struct_span_err(span, format!("non-ASCII character in {desc}"));
            let postfix = if unicode_width::UnicodeWidthChar::width(c).unwrap_or(1) == 0 {
                format!(" but is {c:?}")
            } else {
                String::new()
            };
            err.span_label(span, format!("must be ASCII{postfix}"));
            // Note: the \\xHH suggestions are not given for raw byte string
            // literals, because they are araw and so cannot use any escapes.
            if (c as u32) <= 0xFF && mode != Mode::RawByteStr {
                err.span_suggestion(
                    span,
                    format!(
                        "if you meant to use the unicode code point for {c:?}, use a \\xHH escape"
                    ),
                    format!("\\x{:X}", c as u32),
                    Applicability::MaybeIncorrect,
                );
            } else if mode == Mode::Byte {
                err.span_label(span, "this multibyte character does not fit into a single byte");
            } else if mode != Mode::RawByteStr {
                let mut utf8 = String::new();
                utf8.push(c);
                err.span_suggestion(
                    span,
                    format!("if you meant to use the UTF-8 encoding of {c:?}, use \\xHH escapes"),
                    utf8.as_bytes()
                        .iter()
                        .map(|b: &u8| format!("\\x{:X}", *b))
                        .fold("".to_string(), |a, c| a + &c),
                    Applicability::MaybeIncorrect,
                );
            }
            err.emit()
        }
        EscapeError::OutOfRangeHexEscape => {
            dcx.emit_err(UnescapeError::OutOfRangeHexEscape(err_span))
        }
        EscapeError::LeadingUnderscoreUnicodeEscape => {
            let (c, span) = last_char();
            dcx.emit_err(UnescapeError::LeadingUnderscoreUnicodeEscape {
                span,
                ch: escaped_char(c),
            })
        }
        EscapeError::OverlongUnicodeEscape => {
            dcx.emit_err(UnescapeError::OverlongUnicodeEscape(err_span))
        }
        EscapeError::UnclosedUnicodeEscape => {
            dcx.emit_err(UnescapeError::UnclosedUnicodeEscape(err_span, err_span.shrink_to_hi()))
        }
        EscapeError::NoBraceInUnicodeEscape => {
            let mut suggestion = "\\u{".to_owned();
            let mut suggestion_len = 0;
            let (c, char_span) = last_char();
            let chars = once(c).chain(lit[range.end..].chars());
            for c in chars.take(6).take_while(|c| c.is_digit(16)) {
                suggestion.push(c);
                suggestion_len += c.len_utf8();
            }

            let (label, sub) = if suggestion_len > 0 {
                suggestion.push('}');
                let hi = char_span.lo() + BytePos(suggestion_len as u32);
                (None, NoBraceUnicodeSub::Suggestion { span: err_span.with_hi(hi), suggestion })
            } else {
                (Some(err_span), NoBraceUnicodeSub::Help)
            };
            dcx.emit_err(UnescapeError::NoBraceInUnicodeEscape { span: err_span, label, sub })
        }
        EscapeError::UnicodeEscapeInByte => {
            dcx.emit_err(UnescapeError::UnicodeEscapeInByte(err_span))
        }
        EscapeError::EmptyUnicodeEscape => {
            dcx.emit_err(UnescapeError::EmptyUnicodeEscape(err_span))
        }
        EscapeError::ZeroChars => dcx.emit_err(UnescapeError::ZeroChars(err_span)),
        EscapeError::LoneSlash => dcx.emit_err(UnescapeError::LoneSlash(err_span)),
        EscapeError::NulInCStr => dcx.emit_err(UnescapeError::NulInCStr { span: err_span }),
        EscapeError::UnskippedWhitespaceWarning => {
            let (c, char_span) = last_char();
            dcx.emit_warn(UnescapeError::UnskippedWhitespace {
                span: err_span,
                ch: escaped_char(c),
                char_span,
            });
            return None;
        }
        EscapeError::MultipleSkippedLinesWarning => {
            dcx.emit_warn(UnescapeError::MultipleSkippedLinesWarning(err_span));
            return None;
        }
    })
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
