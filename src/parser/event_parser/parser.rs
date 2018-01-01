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
        assert!(self.pos == self.non_ws_tokens.len());
        self.events
    }

    pub(crate) fn start(&mut self, kind: SyntaxKind) {
        self.event(Event::Start { kind });
    }

    pub(crate) fn finish(&mut self) {
        self.event(Event::Finish);
    }

    fn event(&mut self, event: Event) {
        self.events.push(event)
    }
}