use {Token, SyntaxKind, TextUnit};
use super::Event;
use syntax_kinds::{WHITESPACE, COMMENT};

pub struct Parser<'t> {
    text: &'t str,
    raw_tokens: &'t [Token],
    non_ws_tokens: Vec<(usize, TextUnit)>,

    pos: usize,
    events: Vec<Event>,
}

impl<'t> Parser<'t> {
    pub(crate) fn new(text: &'t str, raw_tokens: &'t [Token]) -> Parser<'t> {
        let mut non_ws_tokens = Vec::new();
        let mut len = TextUnit::new(0);
        for (idx, &token) in raw_tokens.iter().enumerate() {
            match token.kind {
                WHITESPACE | COMMENT => (),
                _ => non_ws_tokens.push((idx, len)),
            }
            len += token.len;
        }

        Parser {
            text,
            raw_tokens,
            non_ws_tokens,

            pos: 0,
            events: Vec::new(),
        }
    }

    pub(crate) fn into_events(self) -> Vec<Event> {
        assert!(self.is_eof());
        self.events
    }

    pub(crate) fn is_eof(&self) -> bool {
        self.pos == self.non_ws_tokens.len()
    }

    pub(crate) fn start(&mut self, kind: SyntaxKind) {
        self.event(Event::Start { kind });
    }

    pub(crate) fn finish(&mut self) {
        self.event(Event::Finish);
    }

    pub(crate) fn bump(&mut self) -> Option<SyntaxKind> {
        if self.is_eof() {
            return None;
        }
        let idx = self.non_ws_tokens[self.pos].0;
        self.pos += 1;
        Some(self.raw_tokens[idx].kind)
    }

    fn event(&mut self, event: Event) {
        self.events.push(event)
    }
}