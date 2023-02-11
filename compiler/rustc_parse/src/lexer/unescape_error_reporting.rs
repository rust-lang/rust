//! Utilities for rendering escape sequence errors as diagnostics.

use std::iter::once;
use std::ops::Range;

use rustc_errors::{Applicability, Handler};
use rustc_lexer::unescape::{EscapeError, Mode};
use rustc_span::{BytePos, Span};

use crate::errors::{MoreThanOneCharNote, MoreThanOneCharSugg, NoBraceUnicodeSub, UnescapeError};

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
    debug!(
        "emit_unescape_error: {:?}, {:?}, {:?}, {:?}, {:?}",
        lit, span_with_quotes, mode, range, error
    );
    let last_char = || {
        let c = lit[range.clone()].chars().rev().next().unwrap();
        let span = span.with_lo(span.hi() - BytePos(c.len_utf8() as u32));
        (c, span)
    };
    match error {
        EscapeError::LoneSurrogateUnicodeEscape => {
            handler.emit_err(UnescapeError::InvalidUnicodeEscape { span, surrogate: true });
        }
        EscapeError::OutOfRangeUnicodeEscape => {
            handler.emit_err(UnescapeError::InvalidUnicodeEscape { span, surrogate: false });
        }
        EscapeError::MoreThanOneChar => {
            use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
            let mut sugg = None;
            let mut note = None;

            let lit_chars = lit.chars().collect::<Vec<_>>();
            let (first, rest) = lit_chars.split_first().unwrap();
            if rest.iter().copied().all(is_combining_mark) {
                let normalized = lit.nfc().to_string();
                if normalized.chars().count() == 1 {
                    let ch = normalized.chars().next().unwrap().escape_default().to_string();
                    sugg = Some(MoreThanOneCharSugg::NormalizedForm { span, ch, normalized });
                }
                let escaped_marks =
                    rest.iter().map(|c| c.escape_default().to_string()).collect::<Vec<_>>();
                note = Some(MoreThanOneCharNote::AllCombining {
                    span,
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
                    sugg =
                        Some(MoreThanOneCharSugg::RemoveNonPrinting { span, ch: ch.to_string() });
                    note = Some(MoreThanOneCharNote::NonPrinting {
                        span,
                        escaped: lit.escape_default().to_string(),
                    });
                }
            };
            let sugg = sugg.unwrap_or_else(|| {
                let is_byte = mode.is_byte();
                let prefix = if is_byte { "b" } else { "" };
                let mut escaped = String::with_capacity(lit.len());
                let mut chrs = lit.chars().peekable();
                while let Some(first) = chrs.next() {
                    match (first, chrs.peek()) {
                        ('\\', Some('"')) => {
                            escaped.push('\\');
                            escaped.push('"');
                            chrs.next();
                        }
                        ('"', _) => {
                            escaped.push('\\');
                            escaped.push('"')
                        }
                        (c, _) => escaped.push(c),
                    };
                }
                let sugg = format!("{prefix}\"{escaped}\"");
                MoreThanOneCharSugg::Quotes { span: span_with_quotes, is_byte, sugg }
            });
            handler.emit_err(UnescapeError::MoreThanOneChar {
                span: span_with_quotes,
                note,
                suggestion: sugg,
            });
        }
        EscapeError::EscapeOnlyChar => {
            let (c, char_span) = last_char();
            handler.emit_err(UnescapeError::EscapeOnlyChar {
                span,
                char_span,
                escaped_sugg: c.escape_default().to_string(),
                escaped_msg: escaped_char(c),
                byte: mode.is_byte(),
            });
        }
        EscapeError::BareCarriageReturn => {
            let double_quotes = mode.in_double_quotes();
            handler.emit_err(UnescapeError::BareCr { span, double_quotes });
        }
        EscapeError::BareCarriageReturnInRawString => {
            assert!(mode.in_double_quotes());
            handler.emit_err(UnescapeError::BareCrRawString(span));
        }
        EscapeError::InvalidEscape => {
            let (c, span) = last_char();

            let label =
                if mode.is_byte() { "unknown byte escape" } else { "unknown character escape" };
            let ec = escaped_char(c);
            let mut diag = handler.struct_span_err(span, &format!("{}: `{}`", label, ec));
            diag.span_label(span, label);
            if c == '{' || c == '}' && !mode.is_byte() {
                diag.help(
                    "if used in a formatting string, curly braces are escaped with `{{` and `}}`",
                );
            } else if c == '\r' {
                diag.help(
                    "this is an isolated carriage return; consider checking your editor and \
                     version control settings",
                );
            } else {
                if !mode.is_byte() {
                    diag.span_suggestion(
                        span_with_quotes,
                        "if you meant to write a literal backslash (perhaps escaping in a regular expression), consider a raw string literal",
                        format!("r\"{}\"", lit),
                        Applicability::MaybeIncorrect,
                    );
                }

                diag.help(
                    "for more information, visit \
                     <https://static.rust-lang.org/doc/master/reference.html#literals>",
                );
            }
            diag.emit();
        }
        EscapeError::TooShortHexEscape => {
            handler.emit_err(UnescapeError::TooShortHexEscape(span));
        }
        EscapeError::InvalidCharInHexEscape | EscapeError::InvalidCharInUnicodeEscape => {
            let (c, span) = last_char();
            let is_hex = error == EscapeError::InvalidCharInHexEscape;
            let ch = escaped_char(c);
            handler.emit_err(UnescapeError::InvalidCharInEscape { span, is_hex, ch });
        }
        EscapeError::NonAsciiCharInByte => {
            let (c, span) = last_char();
            let desc = match mode {
                Mode::Byte => "byte literal",
                Mode::ByteStr => "byte string literal",
                Mode::RawByteStr => "raw byte string literal",
                _ => panic!("non-is_byte literal paired with NonAsciiCharInByte"),
            };
            let mut err = handler.struct_span_err(span, format!("non-ASCII character in {}", desc));
            let postfix = if unicode_width::UnicodeWidthChar::width(c).unwrap_or(1) == 0 {
                format!(" but is {:?}", c)
            } else {
                String::new()
            };
            err.span_label(span, &format!("must be ASCII{}", postfix));
            // Note: the \\xHH suggestions are not given for raw byte string
            // literals, because they are araw and so cannot use any escapes.
            if (c as u32) <= 0xFF && mode != Mode::RawByteStr {
                err.span_suggestion(
                    span,
                    &format!(
                        "if you meant to use the unicode code point for {:?}, use a \\xHH escape",
                        c
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
                    &format!(
                        "if you meant to use the UTF-8 encoding of {:?}, use \\xHH escapes",
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
        EscapeError::OutOfRangeHexEscape => {
            handler.emit_err(UnescapeError::OutOfRangeHexEscape(span));
        }
        EscapeError::LeadingUnderscoreUnicodeEscape => {
            let (c, span) = last_char();
            handler.emit_err(UnescapeError::LeadingUnderscoreUnicodeEscape {
                span,
                ch: escaped_char(c),
            });
        }
        EscapeError::OverlongUnicodeEscape => {
            handler.emit_err(UnescapeError::OverlongUnicodeEscape(span));
        }
        EscapeError::UnclosedUnicodeEscape => {
            handler.emit_err(UnescapeError::UnclosedUnicodeEscape(span, span.shrink_to_hi()));
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
                (None, NoBraceUnicodeSub::Suggestion { span: span.with_hi(hi), suggestion })
            } else {
                (Some(span), NoBraceUnicodeSub::Help)
            };
            handler.emit_err(UnescapeError::NoBraceInUnicodeEscape { span, label, sub });
        }
        EscapeError::UnicodeEscapeInByte => {
            handler.emit_err(UnescapeError::UnicodeEscapeInByte(span));
        }
        EscapeError::EmptyUnicodeEscape => {
            handler.emit_err(UnescapeError::EmptyUnicodeEscape(span));
        }
        EscapeError::ZeroChars => {
            handler.emit_err(UnescapeError::ZeroChars(span));
        }
        EscapeError::LoneSlash => {
            handler.emit_err(UnescapeError::LoneSlash(span));
        }
        EscapeError::UnskippedWhitespaceWarning => {
            let (c, char_span) = last_char();
            handler.emit_warning(UnescapeError::UnskippedWhitespace {
                span,
                ch: escaped_char(c),
                char_span,
            });
        }
        EscapeError::MultipleSkippedLinesWarning => {
            handler.emit_warning(UnescapeError::MultipleSkippedLinesWarning(span));
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
