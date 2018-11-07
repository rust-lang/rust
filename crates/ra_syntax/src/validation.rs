use std::u32;

use arrayvec::ArrayString;

use crate::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast::{self, AstNode},
    SourceFileNode,
    string_lexing::{self, CharComponentKind},
    yellow::{
        SyntaxError,
        SyntaxErrorKind::*,
    },
};

pub(crate) fn validate(file: &SourceFileNode) -> Vec<SyntaxError> {
    let mut errors = Vec::new();
    for node in file.syntax().descendants() {
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
                    assert!(
                        text.chars().count() == 4,
                        "AsciiCodeEscape cannot be longer than 4 chars"
                    );

                    match u8::from_str_radix(&text[2..], 16) {
                        Ok(code) if code < 128 => { /* Escape code is valid */ }
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

                let mut code = ArrayString::<[_; 6]>::new();
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
            }
            CodePoint => {
                // These code points must always be escaped
                if text == "\t" || text == "\r" {
                    errors.push(SyntaxError::new(UnescapedCodepoint, range));
                }
            }
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
        '\\' | '\'' | '"' | 'n' | 'r' | 't' | '0' => true,
        _ => false,
    }
}

#[cfg(test)]
mod test {
    use crate::SourceFileNode;

    fn build_file(literal: &str) -> SourceFileNode {
        let src = format!("const C: char = '{}';", literal);
        SourceFileNode::parse(&src)
    }

    fn assert_valid_char(literal: &str) {
        let file = build_file(literal);
        assert!(
            file.errors().len() == 0,
            "Errors for literal '{}': {:?}",
            literal,
            file.errors()
        );
    }

    fn assert_invalid_char(literal: &str) {
        let file = build_file(literal);
        assert!(file.errors().len() > 0);
    }

    #[test]
    fn test_ansi_codepoints() {
        for byte in 0..=255u8 {
            match byte {
                b'\n' | b'\r' | b'\t' => assert_invalid_char(&(byte as char).to_string()),
                b'\'' | b'\\' => { /* Ignore character close and backslash */ }
                _ => assert_valid_char(&(byte as char).to_string()),
            }
        }
    }

    #[test]
    fn test_unicode_codepoints() {
        let valid = ["Ƒ", "バ", "メ", "﷽"];
        for c in &valid {
            assert_valid_char(c);
        }
    }

    #[test]
    fn test_unicode_multiple_codepoints() {
        let invalid = ["नी", "👨‍👨‍"];
        for c in &invalid {
            assert_invalid_char(c);
        }
    }

    #[test]
    fn test_valid_ascii_escape() {
        let valid = [
            r"\'", "\"", "\\\\", "\\\"", r"\n", r"\r", r"\t", r"\0", "a", "b",
        ];
        for c in &valid {
            assert_valid_char(c);
        }
    }

    #[test]
    fn test_invalid_ascii_escape() {
        let invalid = [r"\a", r"\?", r"\"];
        for c in &invalid {
            assert_invalid_char(c);
        }
    }

    #[test]
    fn test_valid_ascii_code_escape() {
        let valid = [r"\x00", r"\x7F", r"\x55"];
        for c in &valid {
            assert_valid_char(c);
        }
    }

    #[test]
    fn test_invalid_ascii_code_escape() {
        let invalid = [r"\x", r"\x7", r"\xF0"];
        for c in &invalid {
            assert_invalid_char(c);
        }
    }

    #[test]
    fn test_valid_unicode_escape() {
        let valid = [
            r"\u{FF}",
            r"\u{0}",
            r"\u{F}",
            r"\u{10FFFF}",
            r"\u{1_0__FF___FF_____}",
        ];
        for c in &valid {
            assert_valid_char(c);
        }
    }

    #[test]
    fn test_invalid_unicode_escape() {
        let invalid = [
            r"\u",
            r"\u{}",
            r"\u{",
            r"\u{FF",
            r"\u{FFFFFF}",
            r"\u{_F}",
            r"\u{00FFFFF}",
            r"\u{110000}",
        ];
        for c in &invalid {
            assert_invalid_char(c);
        }
    }
}
