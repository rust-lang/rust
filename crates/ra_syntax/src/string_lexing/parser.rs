use rowan::{TextRange, TextUnit};

use self::CharComponentKind::*;

pub struct Parser<'a> {
    pub(super) src: &'a str,
    pub(super) pos: usize,
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

    pub fn skip_whitespace(&mut self) {
        while self.peek().map(|c| c.is_whitespace()) == Some(true) {
            self.advance();
        }
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

    pub fn parse_string_component(&mut self) -> Option<StringComponent> {
        let next = self.peek()?;

        // Ignore string close
        if next == '"' {
            return None;
        }

        let start = self.get_pos();
        self.advance();

        if next == '\\' {
            // Strings can use `\` to ignore newlines, so we first try to parse one of those
            // before falling back to parsing char escapes
            self.parse_ignore_newline(start).or_else(|| {
                let char_component = self.parse_escape(start);
                Some(StringComponent::new(
                    char_component.range,
                    StringComponentKind::Char(char_component.kind),
                ))
            })
        } else {
            let end = self.get_pos();
            Some(StringComponent::new(
                TextRange::from_to(start, end),
                StringComponentKind::Char(CodePoint),
            ))
        }
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
    Char(CharComponentKind),
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
