use crate::{TextRange, TextUnit};
use self::StringComponentKind::*;

#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) struct StringComponent {
    pub(crate) range: TextRange,
    pub(crate) kind: StringComponentKind,
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub(crate) enum StringComponentKind {
    IgnoreNewline,
    CodePoint,
    AsciiEscape,
    AsciiCodeEscape,
    UnicodeEscape,
}

pub(crate) fn parse_quoted_literal(
    prefix: Option<char>,
    quote: char,
    src: &str,
) -> StringComponentIter {
    let prefix = prefix.map(|p| match p {
        'b' => b'b',
        _ => panic!("invalid prefix"),
    });
    let quote = match quote {
        '\'' => b'\'',
        '"' => b'"',
        _ => panic!("invalid quote"),
    };
    StringComponentIter { src, prefix, quote, pos: 0, has_closing_quote: false, suffix: None }
}

pub(crate) struct StringComponentIter<'a> {
    src: &'a str,
    prefix: Option<u8>,
    quote: u8,
    pos: usize,
    pub(crate) has_closing_quote: bool,
    pub(crate) suffix: Option<TextRange>,
}

impl<'a> Iterator for StringComponentIter<'a> {
    type Item = StringComponent;
    fn next(&mut self) -> Option<StringComponent> {
        if self.pos == 0 {
            if let Some(prefix) = self.prefix {
                assert!(
                    self.advance() == prefix as char,
                    "literal should start with a {:?}",
                    prefix as char,
                );
            }
            assert!(
                self.advance() == self.quote as char,
                "literal should start with a {:?}",
                self.quote as char,
            );
        }

        if let Some(component) = self.parse_component() {
            return Some(component);
        }

        // We get here when there are no char components left to parse
        if self.peek() == Some(self.quote as char) {
            self.advance();
            self.has_closing_quote = true;
            if let Some(range) = self.parse_suffix() {
                self.suffix = Some(range);
            }
        }

        assert!(
            self.peek() == None,
            "literal should leave no unparsed input: src = {:?}, pos = {}, length = {}",
            self.src,
            self.pos,
            self.src.len()
        );

        None
    }
}

impl<'a> StringComponentIter<'a> {
    fn peek(&self) -> Option<char> {
        if self.pos == self.src.len() {
            return None;
        }

        self.src[self.pos..].chars().next()
    }

    fn advance(&mut self) -> char {
        let next = self.peek().expect("cannot advance if end of input is reached");
        self.pos += next.len_utf8();
        next
    }

    fn parse_component(&mut self) -> Option<StringComponent> {
        let next = self.peek()?;

        // Ignore string close
        if next == self.quote as char {
            return None;
        }

        let start = self.start_range();
        self.advance();

        if next == '\\' {
            // Strings can use `\` to ignore newlines, so we first try to parse one of those
            // before falling back to parsing char escapes
            if self.quote == b'"' {
                if let Some(component) = self.parse_ignore_newline(start) {
                    return Some(component);
                }
            }

            Some(self.parse_escape(start))
        } else {
            Some(self.finish_component(start, CodePoint))
        }
    }

    fn parse_ignore_newline(&mut self, start: TextUnit) -> Option<StringComponent> {
        // In string literals, when a `\` occurs immediately before the newline, the `\`,
        // the newline, and all whitespace at the beginning of the next line are ignored
        match self.peek() {
            Some('\n') | Some('\r') => {
                self.skip_whitespace();
                Some(self.finish_component(start, IgnoreNewline))
            }
            _ => None,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek().map(|c| c.is_whitespace()) == Some(true) {
            self.advance();
        }
    }

    fn parse_escape(&mut self, start: TextUnit) -> StringComponent {
        if self.peek().is_none() {
            return self.finish_component(start, AsciiEscape);
        }

        let next = self.advance();
        match next {
            'x' => self.parse_ascii_code_escape(start),
            'u' => self.parse_unicode_escape(start),
            _ => self.finish_component(start, AsciiEscape),
        }
    }

    fn parse_unicode_escape(&mut self, start: TextUnit) -> StringComponent {
        match self.peek() {
            Some('{') => {
                self.advance();

                // Parse anything until we reach `}`
                while let Some(next) = self.peek() {
                    self.advance();
                    if next == '}' {
                        break;
                    }
                }

                self.finish_component(start, UnicodeEscape)
            }
            Some(_) | None => self.finish_component(start, UnicodeEscape),
        }
    }

    fn parse_ascii_code_escape(&mut self, start: TextUnit) -> StringComponent {
        let code_start = self.pos;
        while let Some(next) = self.peek() {
            if next == '\'' || (self.pos - code_start == 2) {
                break;
            }

            self.advance();
        }
        self.finish_component(start, AsciiCodeEscape)
    }

    fn parse_suffix(&mut self) -> Option<TextRange> {
        let start = self.start_range();
        let _ = self.peek()?;
        while let Some(_) = self.peek() {
            self.advance();
        }
        Some(self.finish_range(start))
    }

    fn start_range(&self) -> TextUnit {
        TextUnit::from_usize(self.pos)
    }

    fn finish_range(&self, start: TextUnit) -> TextRange {
        TextRange::from_to(start, TextUnit::from_usize(self.pos))
    }

    fn finish_component(&self, start: TextUnit, kind: StringComponentKind) -> StringComponent {
        let range = self.finish_range(start);
        StringComponent { range, kind }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> (bool, Vec<StringComponent>) {
        let component_iterator = &mut parse_quoted_literal(None, '\'', src);
        let components: Vec<_> = component_iterator.collect();
        (component_iterator.has_closing_quote, components)
    }

    fn unclosed_char_component(src: &str) -> StringComponent {
        let (has_closing_quote, components) = parse(src);
        assert!(!has_closing_quote, "char should not have closing quote");
        assert!(components.len() == 1);
        components[0].clone()
    }

    fn closed_char_component(src: &str) -> StringComponent {
        let (has_closing_quote, components) = parse(src);
        assert!(has_closing_quote, "char should have closing quote");
        assert!(components.len() == 1, "Literal: {}\nComponents: {:#?}", src, components);
        components[0].clone()
    }

    fn closed_char_components(src: &str) -> Vec<StringComponent> {
        let (has_closing_quote, components) = parse(src);
        assert!(has_closing_quote, "char should have closing quote");
        components
    }

    fn range_closed(src: &str) -> TextRange {
        TextRange::from_to(1.into(), (src.len() as u32 - 1).into())
    }

    fn range_unclosed(src: &str) -> TextRange {
        TextRange::from_to(1.into(), (src.len() as u32).into())
    }

    #[test]
    fn test_unicode_escapes() {
        let unicode_escapes = &[r"{DEAD}", "{BEEF}", "{FF}", "{}", ""];
        for escape in unicode_escapes {
            let escape_sequence = format!(r"'\u{}'", escape);
            let component = closed_char_component(&escape_sequence);
            let expected_range = range_closed(&escape_sequence);
            assert_eq!(component.kind, UnicodeEscape);
            assert_eq!(component.range, expected_range);
        }
    }

    #[test]
    fn test_unicode_escapes_unclosed() {
        let unicode_escapes = &["{DEAD", "{BEEF", "{FF"];
        for escape in unicode_escapes {
            let escape_sequence = format!(r"'\u{}'", escape);
            let component = unclosed_char_component(&escape_sequence);
            let expected_range = range_unclosed(&escape_sequence);
            assert_eq!(component.kind, UnicodeEscape);
            assert_eq!(component.range, expected_range);
        }
    }

    #[test]
    fn test_empty_char() {
        let (has_closing_quote, components) = parse("''");
        assert!(has_closing_quote, "char should have closing quote");
        assert!(components.len() == 0);
    }

    #[test]
    fn test_unclosed_char() {
        let component = unclosed_char_component("'a");
        assert!(component.kind == CodePoint);
        assert!(component.range == TextRange::from_to(1.into(), 2.into()));
    }

    #[test]
    fn test_digit_escapes() {
        let literals = &[r"", r"5", r"55"];

        for literal in literals {
            let lit_text = format!(r"'\x{}'", literal);
            let component = closed_char_component(&lit_text);
            assert!(component.kind == AsciiCodeEscape);
            assert!(component.range == range_closed(&lit_text));
        }

        // More than 2 digits starts a new codepoint
        let components = closed_char_components(r"'\x555'");
        assert!(components.len() == 2);
        assert!(components[1].kind == CodePoint);
    }

    #[test]
    fn test_ascii_escapes() {
        let literals = &[
            r"\'", "\\\"", // equivalent to \"
            r"\n", r"\r", r"\t", r"\\", r"\0",
        ];

        for literal in literals {
            let lit_text = format!("'{}'", literal);
            let component = closed_char_component(&lit_text);
            assert!(component.kind == AsciiEscape);
            assert!(component.range == range_closed(&lit_text));
        }
    }

    #[test]
    fn test_no_escapes() {
        let literals = &['"', 'n', 'r', 't', '0', 'x', 'u'];

        for &literal in literals {
            let lit_text = format!("'{}'", literal);
            let component = closed_char_component(&lit_text);
            assert!(component.kind == CodePoint);
            assert!(component.range == range_closed(&lit_text));
        }
    }
}
