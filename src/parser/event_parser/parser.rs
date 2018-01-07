use {Token, SyntaxKind, TextUnit};
use super::{Event};
use super::super::is_insignificant;
use syntax_kinds::{L_CURLY, R_CURLY, ERROR};
use tree::EOF;


pub(crate) struct Parser<'t> {
    #[allow(unused)]
    text: &'t str,
    #[allow(unused)]
    start_offsets: Vec<TextUnit>,
    tokens: Vec<Token>, // non-whitespace tokens

    pos: usize,
    events: Vec<Event>,

    curly_level: i32,
    curly_limit: Option<i32>,
}

impl<'t> Parser<'t> {
    pub(crate) fn new(text: &'t str, raw_tokens: &'t [Token]) -> Parser<'t> {
        let mut tokens = Vec::new();
        let mut start_offsets = Vec::new();
        let mut len = TextUnit::new(0);
        for &token in raw_tokens.iter() {
            if !is_insignificant(token.kind) {
                tokens.push(token);
                start_offsets.push(len);
            }
            len += token.len;
        }

        Parser {
            text,
            start_offsets,
            tokens,

            pos: 0,
            events: Vec::new(),
            curly_level: 0,
            curly_limit: None,
        }
    }

    pub(crate) fn into_events(self) -> Vec<Event> {
        assert!(self.curly_limit.is_none());
        assert!(self.current() == EOF);
        self.events
    }

    pub(crate) fn current(&self) -> SyntaxKind {
        if self.pos == self.tokens.len() {
            return EOF;
        }
        let token = self.tokens[self.pos];
        if let Some(limit) = self.curly_limit {
            if limit == self.curly_level && token.kind == R_CURLY {
                return EOF
            }
        }
        token.kind
    }

    pub(crate) fn start(&mut self, kind: SyntaxKind) {
        self.event(Event::Start { kind });
    }

    pub(crate) fn finish(&mut self) {
        self.event(Event::Finish);
    }

    pub(crate) fn error<'p>(&'p mut self) -> ErrorBuilder<'p, 't> {
        ErrorBuilder::new(self)
    }

    pub(crate) fn bump(&mut self) -> SyntaxKind {
        let kind = self.current();
        match kind {
            L_CURLY => self.curly_level += 1,
            R_CURLY => self.curly_level -= 1,
            EOF => return EOF,
            _ => (),
        }
        self.pos += 1;
        self.event(Event::Token { kind, n_raw_tokens: 1 });
        kind
    }

    pub(crate) fn lookahead(&self, kinds: &[SyntaxKind]) -> bool {
        if self.tokens[self.pos..].len() < kinds.len() {
            return false
        }
        kinds.iter().zip(self.tokens[self.pos..].iter().map(|t| t.kind))
            .all(|(&k1, k2)| k1 == k2)
    }

    pub(crate) fn curly_block<F: FnOnce(&mut Parser)>(&mut self, f: F) -> bool {
        let old_level = self.curly_level;
        let old_limit = self.curly_limit;
        if !self.expect(L_CURLY) {
            return false
        }
        self.curly_limit = Some(self.curly_level);
        f(self);
        assert!(self.curly_level > old_level);
        self.curly_limit = old_limit;
        if !self.expect(R_CURLY) {
            self.start(ERROR);
            while self.curly_level > old_level {
                if self.bump() == EOF {
                    break;
                }
            }
            self.finish();
        }
        true
    }

    fn event(&mut self, event: Event) {
        self.events.push(event)
    }
}

pub(crate) struct ErrorBuilder<'p, 't: 'p> {
    message: Option<String>,
    parser: &'p mut Parser<'t>
}

impl<'t, 'p> ErrorBuilder<'p, 't> {
    fn new(parser: &'p mut Parser<'t>) -> Self {
        ErrorBuilder { message: None, parser }
    }

    pub fn message<M: Into<String>>(mut self, m: M) -> Self {
        self.message = Some(m.into());
        self
    }

    pub fn emit(self) {
        let message = self.message.expect("Error message not set");
        self.parser.event(Event::Error { message });
    }
}
