use {Token, SyntaxKind, TextUnit};
use super::{Event};
use super::super::is_insignificant;
use syntax_kinds::{L_CURLY, R_CURLY, ERROR};

pub struct Parser<'t> {
    text: &'t str,
    raw_tokens: &'t [Token],
    non_ws_tokens: Vec<(usize, TextUnit)>,

    pos: usize,
    events: Vec<Event>,

    curly_level: i32,
    curly_limit: Option<i32>,
}

impl<'t> Parser<'t> {
    pub(crate) fn new(text: &'t str, raw_tokens: &'t [Token]) -> Parser<'t> {
        let mut non_ws_tokens = Vec::new();
        let mut len = TextUnit::new(0);
        for (idx, &token) in raw_tokens.iter().enumerate() {
            if !is_insignificant(token.kind) {
                non_ws_tokens.push((idx, len))
            }
            len += token.len;
        }

        Parser {
            text,
            raw_tokens,
            non_ws_tokens,

            pos: 0,
            events: Vec::new(),
            curly_level: 0,
            curly_limit: None,
        }
    }

    pub(crate) fn into_events(self) -> Vec<Event> {
        assert!(self.is_eof());
        self.events
    }

    pub(crate) fn is_eof(&self) -> bool {
        if self.pos == self.non_ws_tokens.len() {
            return true
        }
        if let Some(limit) = self.curly_limit {
            let idx = self.non_ws_tokens[self.pos].0;
            return limit == self.curly_level && self.raw_tokens[idx].kind == R_CURLY;
        }
        false
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

    pub(crate) fn current(&self) -> Option<SyntaxKind> {
        if self.is_eof() {
            return None;
        }
        let idx = self.non_ws_tokens[self.pos].0;
        Some(self.raw_tokens[idx].kind)
    }

    pub(crate) fn bump(&mut self) -> Option<SyntaxKind> {
        let kind = self.current()?;
        match kind {
            L_CURLY => self.curly_level += 1,
            R_CURLY => self.curly_level -= 1,
            _ => (),
        }
        self.pos += 1;
        self.event(Event::Token { kind, n_raw_tokens: 1 });
        Some(kind)
    }

    pub(crate) fn lookahead(&self, kinds: &[SyntaxKind]) -> bool {
        if self.non_ws_tokens[self.pos..].len() < kinds.len() {
            return false
        }
        kinds.iter().zip(self.non_ws_tokens[self.pos..].iter())
            .all(|(&k1, &(idx, _))| k1 == self.raw_tokens[idx].kind)
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
                if self.bump().is_none() {
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
