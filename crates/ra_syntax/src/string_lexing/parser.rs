use rowan::{TextRange, TextUnit};

use self::StringComponentKind::*;

pub struct Parser<'a> {
    pub(super) quote: u8,
    pub(super) src: &'a str,
    pub(super) pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str, quote: u8) -> Parser<'a> {
        Parser { quote, src, pos: 0 }
    }

    // Utility methods

    pub fn peek(&self) -> Option<char> {
        if self.pos == self.src.len() {
            return None;
        }

        self.src[self.pos..].chars().next()
    }

    pub fn advance(&mut self) -> char {
        let next = self.peek().expect("cannot advance if end of input is reached");
        self.pos += next.len_utf8();
        next
    }

    pub fn skip_whitespace(&mut self) {
        while self.peek().map(|c| c.is_whitespace()) == Some(true) {
            self.advance();
        }
    }

    pub fn get_pos(&self) -> TextUnit {
        (self.pos as u32).into()
    }

    // Char parsing methods

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

                let end = self.get_pos();
                StringComponent::new(TextRange::from_to(start, end), UnicodeEscape)
            }
            Some(_) | None => {
                let end = self.get_pos();
                StringComponent::new(TextRange::from_to(start, end), UnicodeEscape)
            }
        }
    }

    fn parse_ascii_code_escape(&mut self, start: TextUnit) -> StringComponent {
        let code_start = self.get_pos();
        while let Some(next) = self.peek() {
            if next == '\'' || (self.get_pos() - code_start == 2.into()) {
                break;
            }

            self.advance();
        }

        let end = self.get_pos();
        StringComponent::new(TextRange::from_to(start, end), AsciiCodeEscape)
    }

    fn parse_escape(&mut self, start: TextUnit) -> StringComponent {
        if self.peek().is_none() {
            return StringComponent::new(TextRange::from_to(start, self.get_pos()), AsciiEscape);
        }

        let next = self.advance();
        let end = self.get_pos();
        let range = TextRange::from_to(start, end);
        match next {
            'x' => self.parse_ascii_code_escape(start),
            'u' => self.parse_unicode_escape(start),
            _ => StringComponent::new(range, AsciiEscape),
        }
    }

    pub fn parse_ignore_newline(&mut self, start: TextUnit) -> Option<StringComponent> {
        // In string literals, when a `\` occurs immediately before the newline, the `\`,
        // the newline, and all whitespace at the beginning of the next line are ignored
        match self.peek() {
            Some('\n') | Some('\r') => {
                self.skip_whitespace();
                Some(StringComponent::new(
                    TextRange::from_to(start, self.get_pos()),
                    StringComponentKind::IgnoreNewline,
                ))
            }
            _ => None,
        }
    }

    pub fn parse_component(&mut self) -> Option<StringComponent> {
        let next = self.peek()?;

        // Ignore string close
        if next == self.quote as char {
            return None;
        }

        let start = self.get_pos();
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
            let end = self.get_pos();
            Some(StringComponent::new(TextRange::from_to(start, end), CodePoint))
        }
    }

    pub fn parse_suffix(&mut self) -> Option<TextRange> {
        let start = self.get_pos();
        let _ = self.peek()?;
        while let Some(_) = self.peek() {
            self.advance();
        }
        let end = self.get_pos();
        Some(TextRange::from_to(start, end))
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct StringComponent {
    pub range: TextRange,
    pub kind: StringComponentKind,
}

impl StringComponent {
    fn new(range: TextRange, kind: StringComponentKind) -> StringComponent {
        StringComponent { range, kind }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum StringComponentKind {
    IgnoreNewline,
    CodePoint,
    AsciiEscape,
    AsciiCodeEscape,
    UnicodeEscape,
}
