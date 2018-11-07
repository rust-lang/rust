use std::u32;

use crate::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast::{self, AstNode},
    File,
    string_lexing::{self, CharComponentKind},
    utils::MutAsciiString,
    yellow::{
        SyntaxError,
        SyntaxErrorKind::*,
    },
};

pub(crate) fn validate(file: &File) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.root.borrowed().descendants() {
        let _ = visitor_ctx(&mut errors)
            .visit::<ast::Char, _>(validate_char)
            .accept(node);
    }
    errors
}

fn validate_char(node: ast::Char, errors: &mut Vec<SyntaxError>) {
    let mut components = string_lexing::parse_char_literal(node.text());
    let mut len = 0;
    for component in &mut components {
        len += 1;

        // Validate escapes
        let text = &node.text()[component.range];
        let range = component.range + node.syntax().range().start();
        use self::CharComponentKind::*;
        match component.kind {
            AsciiEscape => {
                if text.len() == 1 {
                    // Escape sequence consists only of leading `\`
                    errors.push(SyntaxError::new(EmptyAsciiEscape, range));
                } else {
                    let escape_code = text.chars().skip(1).next().unwrap();
                    if !is_ascii_escape(escape_code) {
                        errors.push(SyntaxError::new(InvalidAsciiEscape, range));
                    }
                }
            }
            AsciiCodeEscape => {
                // An AsciiCodeEscape has 4 chars, example: `\xDD`
                if text.len() < 4 {
                    errors.push(SyntaxError::new(TooShortAsciiCodeEscape, range));
                } else {
                    assert!(text.chars().count() == 4, "AsciiCodeEscape cannot be longer than 4 chars");

                    match u8::from_str_radix(&text[2..], 16) {
                        Ok(code) if code < 128 => { /* Escape code is valid */ },
                        Ok(_) => errors.push(SyntaxError::new(AsciiCodeEscapeOutOfRange, range)),
                        Err(_) => errors.push(SyntaxError::new(MalformedAsciiCodeEscape, range)),
                    }

                }
            }
            UnicodeEscape => {
                assert!(&text[..2] == "\\u", "UnicodeEscape always starts with \\u");

                if text.len() == 2 {
                    // No starting `{`
                    errors.push(SyntaxError::new(MalformedUnicodeEscape, range));
                    return;
                }

                if text.len() == 3 {
                    // Only starting `{`
                    errors.push(SyntaxError::new(UnclosedUnicodeEscape, range));
                    return;
                }

                let mut buf = &mut [0; 6];
                let mut code = MutAsciiString::new(buf);
                let mut closed = false;
                for c in text[3..].chars() {
                    assert!(!closed, "no characters after escape is closed");

                    if c.is_digit(16) {
                        if code.len() == 6 {
                            errors.push(SyntaxError::new(OverlongUnicodeEscape, range));
                            return;
                        }

                        code.push(c);
                    } else if c == '_' {
                        // Reject leading _
                        if code.len() == 0 {
                            errors.push(SyntaxError::new(MalformedUnicodeEscape, range));
                            return;
                        }
                    } else if c == '}' {
                        closed = true;
                    } else {
                        errors.push(SyntaxError::new(MalformedUnicodeEscape, range));
                        return;
                    }
                }

                if !closed {
                    errors.push(SyntaxError::new(UnclosedUnicodeEscape, range))
                }

                if code.len() == 0 {
                    errors.push(SyntaxError::new(EmptyUnicodeEcape, range));
                    return;
                }

                match u32::from_str_radix(&code, 16) {
                    Ok(code_u32) if code_u32 > 0x10FFFF => {
                        errors.push(SyntaxError::new(UnicodeEscapeOutOfRange, range));
                    }
                    Ok(_) => {
                        // Valid escape code
                    }
                    Err(_) => {
                        errors.push(SyntaxError::new(MalformedUnicodeEscape, range));
                    }
                }

                // FIXME: we really need tests for this
            }
            // Code points are always valid
            CodePoint => (),
        }
    }

    if !components.has_closing_quote {
        errors.push(SyntaxError::new(UnclosedChar, node.syntax().range()));
    }

    if len == 0 {
        errors.push(SyntaxError::new(EmptyChar, node.syntax().range()));
    }

    if len > 1 {
        errors.push(SyntaxError::new(LongChar, node.syntax().range()));
    }
}

fn is_ascii_escape(code: char) -> bool {
    match code {
        '\'' | '"' | 'n' | 'r' | 't' | '0' => true,
        _ => false,
    }
}
