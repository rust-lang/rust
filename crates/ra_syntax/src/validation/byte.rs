//! Validation of byte literals

use crate::{
    ast::{self, AstNode},
    string_lexing::{self, StringComponentKind},
    TextRange,
    validation::char,
    yellow::{
        SyntaxError,
        SyntaxErrorKind::*,
    },
};

pub(super) fn validate_byte_node(node: ast::Byte, errors: &mut Vec<SyntaxError>) {
    let literal_text = node.text();
    let literal_range = node.syntax().range();
    let mut components = string_lexing::parse_byte_literal(literal_text);
    let mut len = 0;
    for component in &mut components {
        len += 1;
        let text = &literal_text[component.range];
        let range = component.range + literal_range.start();
        validate_byte_component(text, component.kind, range, errors);
    }

    if !components.has_closing_quote {
        errors.push(SyntaxError::new(UnclosedByte, literal_range));
    }

    if let Some(range) = components.suffix {
        errors.push(SyntaxError::new(
            InvalidSuffix,
            range + literal_range.start(),
        ));
    }

    if len == 0 {
        errors.push(SyntaxError::new(EmptyByte, literal_range));
    }

    if len > 1 {
        errors.push(SyntaxError::new(OverlongByte, literal_range));
    }
}

pub(super) fn validate_byte_component(
    text: &str,
    kind: StringComponentKind,
    range: TextRange,
    errors: &mut Vec<SyntaxError>,
) {
    use self::StringComponentKind::*;
    match kind {
        AsciiEscape => validate_byte_escape(text, range, errors),
        AsciiCodeEscape => validate_byte_code_escape(text, range, errors),
        UnicodeEscape => errors.push(SyntaxError::new(UnicodeEscapeForbidden, range)),
        CodePoint => {
            let c = text
                .chars()
                .next()
                .expect("Code points should be one character long");

            // These bytes must always be escaped
            if c == '\t' || c == '\r' || c == '\n' {
                errors.push(SyntaxError::new(UnescapedByte, range));
            }

            // Only ASCII bytes are allowed
            if c > 0x7F as char {
                errors.push(SyntaxError::new(ByteOutOfRange, range));
            }
        }
        IgnoreNewline => { /* always valid */ }
    }
}

fn validate_byte_escape(text: &str, range: TextRange, errors: &mut Vec<SyntaxError>) {
    if text.len() == 1 {
        // Escape sequence consists only of leading `\`
        errors.push(SyntaxError::new(EmptyByteEscape, range));
    } else {
        let escape_code = text.chars().skip(1).next().unwrap();
        if !char::is_ascii_escape(escape_code) {
            errors.push(SyntaxError::new(InvalidByteEscape, range));
        }
    }
}

fn validate_byte_code_escape(text: &str, range: TextRange, errors: &mut Vec<SyntaxError>) {
    // A ByteCodeEscape has 4 chars, example: `\xDD`
    if text.len() < 4 {
        errors.push(SyntaxError::new(TooShortByteCodeEscape, range));
    } else {
        assert!(
            text.chars().count() == 4,
            "ByteCodeEscape cannot be longer than 4 chars"
        );

        if u8::from_str_radix(&text[2..], 16).is_err() {
            errors.push(SyntaxError::new(MalformedByteCodeEscape, range));
        }
    }
}

#[cfg(test)]
mod test {
    use crate::SourceFileNode;

    fn build_file(literal: &str) -> SourceFileNode {
        let src = format!("const C: u8 = b'{}';", literal);
        SourceFileNode::parse(&src)
    }

    fn assert_valid_byte(literal: &str) {
        let file = build_file(literal);
        assert!(
            file.errors().len() == 0,
            "Errors for literal '{}': {:?}",
            literal,
            file.errors()
        );
    }

    fn assert_invalid_byte(literal: &str) {
        let file = build_file(literal);
        assert!(file.errors().len() > 0);
    }

    #[test]
    fn test_ansi_codepoints() {
        for byte in 0..128 {
            match byte {
                b'\n' | b'\r' | b'\t' => assert_invalid_byte(&(byte as char).to_string()),
                b'\'' | b'\\' => { /* Ignore character close and backslash */ }
                _ => assert_valid_byte(&(byte as char).to_string()),
            }
        }

        for byte in 128..=255u8 {
            assert_invalid_byte(&(byte as char).to_string());
        }
    }

    #[test]
    fn test_unicode_codepoints() {
        let invalid = ["Ƒ", "バ", "メ", "﷽"];
        for c in &invalid {
            assert_invalid_byte(c);
        }
    }

    #[test]
    fn test_unicode_multiple_codepoints() {
        let invalid = ["नी", "👨‍👨‍"];
        for c in &invalid {
            assert_invalid_byte(c);
        }
    }

    #[test]
    fn test_valid_byte_escape() {
        let valid = [r"\'", "\"", "\\\\", "\\\"", r"\n", r"\r", r"\t", r"\0"];
        for c in &valid {
            assert_valid_byte(c);
        }
    }

    #[test]
    fn test_invalid_byte_escape() {
        let invalid = [r"\a", r"\?", r"\"];
        for c in &invalid {
            assert_invalid_byte(c);
        }
    }

    #[test]
    fn test_valid_byte_code_escape() {
        let valid = [r"\x00", r"\x7F", r"\x55", r"\xF0"];
        for c in &valid {
            assert_valid_byte(c);
        }
    }

    #[test]
    fn test_invalid_byte_code_escape() {
        let invalid = [r"\x", r"\x7"];
        for c in &invalid {
            assert_invalid_byte(c);
        }
    }

    #[test]
    fn test_invalid_unicode_escape() {
        let well_formed = [
            r"\u{FF}",
            r"\u{0}",
            r"\u{F}",
            r"\u{10FFFF}",
            r"\u{1_0__FF___FF_____}",
        ];
        for c in &well_formed {
            assert_invalid_byte(c);
        }

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
            assert_invalid_byte(c);
        }
    }
}
