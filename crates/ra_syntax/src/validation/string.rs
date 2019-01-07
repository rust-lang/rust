use crate::{
    ast::{self, AstNode},
    string_lexing,
    yellow::{
        SyntaxError,
        SyntaxErrorKind::*,
    },
};

use super::char;

pub(crate) fn validate_string_node(node: &ast::String, errors: &mut Vec<SyntaxError>) {
    let literal_text = node.text();
    let literal_range = node.syntax().range();
    let mut components = string_lexing::parse_string_literal(literal_text);
    for component in &mut components {
        let range = component.range + literal_range.start();

        // Chars must escape \t, \n and \r codepoints, but strings don't
        let text = &literal_text[component.range];
        match text {
            "\t" | "\n" | "\r" => { /* always valid */ }
            _ => char::validate_char_component(text, component.kind, range, errors),
        }
    }

    if !components.has_closing_quote {
        errors.push(SyntaxError::new(UnclosedString, literal_range));
    }

    if let Some(range) = components.suffix {
        errors.push(SyntaxError::new(
            InvalidSuffix,
            range + literal_range.start(),
        ));
    }
}

#[cfg(test)]
mod test {
    use crate::{SourceFile, TreePtr};

    fn build_file(literal: &str) -> TreePtr<SourceFile> {
        let src = format!(r#"const S: &'static str = "{}";"#, literal);
        println!("Source: {}", src);
        SourceFile::parse(&src)
    }

    fn assert_valid_str(literal: &str) {
        let file = build_file(literal);
        assert!(
            file.errors().len() == 0,
            "Errors for literal '{}': {:?}",
            literal,
            file.errors()
        );
    }

    fn assert_invalid_str(literal: &str) {
        let file = build_file(literal);
        assert!(file.errors().len() > 0);
    }

    #[test]
    fn test_ansi_codepoints() {
        for byte in 0..=255u8 {
            match byte {
                b'\"' | b'\\' => { /* Ignore string close and backslash */ }
                _ => assert_valid_str(&(byte as char).to_string()),
            }
        }
    }

    #[test]
    fn test_unicode_codepoints() {
        let valid = ["Ƒ", "バ", "メ", "﷽"];
        for c in &valid {
            assert_valid_str(c);
        }
    }

    #[test]
    fn test_unicode_multiple_codepoints() {
        let valid = ["नी", "👨‍👨‍"];
        for c in &valid {
            assert_valid_str(c);
        }
    }

    #[test]
    fn test_valid_ascii_escape() {
        let valid = [r"\'", r#"\""#, r"\\", r"\n", r"\r", r"\t", r"\0", "a", "b"];
        for c in &valid {
            assert_valid_str(c);
        }
    }

    #[test]
    fn test_invalid_ascii_escape() {
        let invalid = [r"\a", r"\?", r"\"];
        for c in &invalid {
            assert_invalid_str(c);
        }
    }

    #[test]
    fn test_valid_ascii_code_escape() {
        let valid = [r"\x00", r"\x7F", r"\x55"];
        for c in &valid {
            assert_valid_str(c);
        }
    }

    #[test]
    fn test_invalid_ascii_code_escape() {
        let invalid = [r"\x", r"\x7", r"\xF0"];
        for c in &invalid {
            assert_invalid_str(c);
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
            assert_valid_str(c);
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
            assert_invalid_str(c);
        }
    }

    #[test]
    fn test_mixed() {
        assert_valid_str(
            r"This is the tale of a string
with a newline in between, some emoji (👨‍👨‍) here and there,
unicode escapes like this: \u{1FFBB} and weird stuff like
this ﷽",
        );
    }

    #[test]
    fn test_ignore_newline() {
        assert_valid_str(
            "Hello \
             World",
        );
    }
}
