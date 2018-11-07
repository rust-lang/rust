use self::CharComponentKind::*;
use rowan::{TextRange, TextUnit};

pub fn parse_char_literal(src: &str) -> CharComponentIterator {
    CharComponentIterator {
        parser: Parser::new(src),
        has_closing_quote: false,
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CharComponent {
    pub range: TextRange,
    pub kind: CharComponentKind,
}

impl CharComponent {
    fn new(range: TextRange, kind: CharComponentKind) -> CharComponent {
        CharComponent { range, kind }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum CharComponentKind {
    CodePoint,
    AsciiEscape,
    AsciiCodeEscape,
    UnicodeEscape,
}

pub struct CharComponentIterator<'a> {
    parser: Parser<'a>,
    pub has_closing_quote: bool,
}

impl<'a> Iterator for CharComponentIterator<'a> {
    type Item = CharComponent;
    fn next(&mut self) -> Option<CharComponent> {
        if self.parser.pos == 0 {
            assert!(
                self.parser.advance() == '\'',
                "char literal should start with a quote"
            );
        }

        if let Some(component) = self.parser.parse_char_component() {
            return Some(component);
        }

        // We get here when there are no char components left to parse
        if self.parser.peek() == Some('\'') {
            self.parser.advance();
            self.has_closing_quote = true;
        }

        assert!(
            self.parser.peek() == None,
            "char literal should leave no unparsed input: src = {}, pos = {}, length = {}",
            self.parser.src,
            self.parser.pos,
            self.parser.src.len()
        );

        None
    }
}

pub struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str) -> Parser<'a> {
        Parser { src, pos: 0 }
    }

    // Utility methods

    pub fn peek(&self) -> Option<char> {
        if self.pos == self.src.len() {
            return None;
        }

        self.src[self.pos..].chars().next()
    }

    pub fn advance(&mut self) -> char {
        let next = self
            .peek()
            .expect("cannot advance if end of input is reached");
        self.pos += next.len_utf8();
        next
    }

    pub fn get_pos(&self) -> TextUnit {
        (self.pos as u32).into()
    }

    // Char parsing methods

    fn parse_unicode_escape(&mut self, start: TextUnit) -> CharComponent {
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

                let end = self.get_pos();
                CharComponent::new(TextRange::from_to(start, end), UnicodeEscape)
            }
            Some(_) | None => {
                let end = self.get_pos();
                CharComponent::new(TextRange::from_to(start, end), UnicodeEscape)
            }
        }
    }

    fn parse_ascii_code_escape(&mut self, start: TextUnit) -> CharComponent {
        let code_start = self.get_pos();
        while let Some(next) = self.peek() {
            if next == '\'' || (self.get_pos() - code_start == 2.into()) {
                break;
            }

            self.advance();
        }

        let end = self.get_pos();
        CharComponent::new(TextRange::from_to(start, end), AsciiCodeEscape)
    }

    fn parse_escape(&mut self, start: TextUnit) -> CharComponent {
        if self.peek().is_none() {
            return CharComponent::new(TextRange::from_to(start, start), AsciiEscape);
        }

        let next = self.advance();
        let end = self.get_pos();
        let range = TextRange::from_to(start, end);
        match next {
            'x' => self.parse_ascii_code_escape(start),
            'u' => self.parse_unicode_escape(start),
            _ => CharComponent::new(range, AsciiEscape),
        }
    }

    pub fn parse_char_component(&mut self) -> Option<CharComponent> {
        let next = self.peek()?;

        // Ignore character close
        if next == '\'' {
            return None;
        }

        let start = self.get_pos();
        self.advance();

        if next == '\\' {
            Some(self.parse_escape(start))
        } else {
            let end = self.get_pos();
            Some(CharComponent::new(
                TextRange::from_to(start, end),
                CodePoint,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> (bool, Vec<CharComponent>) {
        let component_iterator = &mut super::parse_char_literal(src);
        let components: Vec<_> = component_iterator.collect();
        (component_iterator.has_closing_quote, components)
    }

    fn unclosed_char_component(src: &str) -> CharComponent {
        let (has_closing_quote, components) = parse(src);
        assert!(!has_closing_quote, "char should not have closing quote");
        assert!(components.len() == 1);
        components[0].clone()
    }

    fn closed_char_component(src: &str) -> CharComponent {
        let (has_closing_quote, components) = parse(src);
        assert!(has_closing_quote, "char should have closing quote");
        assert!(
            components.len() == 1,
            "Literal: {}\nComponents: {:#?}",
            src,
            components
        );
        components[0].clone()
    }

    fn closed_char_components(src: &str) -> Vec<CharComponent> {
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
            assert_eq!(component.kind, CharComponentKind::UnicodeEscape);
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
            assert_eq!(component.kind, CharComponentKind::UnicodeEscape);
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
            assert!(component.kind == CharComponentKind::AsciiCodeEscape);
            assert!(component.range == range_closed(&lit_text));
        }

        // More than 2 digits starts a new codepoint
        let components = closed_char_components(r"'\x555'");
        assert!(components.len() == 2);
        assert!(components[1].kind == CharComponentKind::CodePoint);
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
            assert!(component.kind == CharComponentKind::AsciiEscape);
            assert!(component.range == range_closed(&lit_text));
        }
    }

    #[test]
    fn test_no_escapes() {
        let literals = &['"', 'n', 'r', 't', '0', 'x', 'u'];

        for &literal in literals {
            let lit_text = format!("'{}'", literal);
            let component = closed_char_component(&lit_text);
            assert!(component.kind == CharComponentKind::CodePoint);
            assert!(component.range == range_closed(&lit_text));
        }
    }
}
