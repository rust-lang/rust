mod event;
mod input;

use std::cell::Cell;

use crate::{
    lexer::Token,
    parser_api::Parser,
    parser_impl::{
        event::{Event, EventProcessor},
        input::{InputPosition, ParserInput},
    },
    SmolStr,
    syntax_node::syntax_error::{
        ParseError,
        SyntaxError,
    },
};

use crate::SyntaxKind::{self, EOF, TOMBSTONE};

pub(crate) trait Sink {
    type Tree;

    /// Adds new leaf to the current branch.
    fn leaf(&mut self, kind: SyntaxKind, text: SmolStr);

    /// Start new branch and make it current.
    fn start_branch(&mut self, kind: SyntaxKind);

    /// Finish current branch and restore previous
    /// branch as current.
    fn finish_branch(&mut self);

    fn error(&mut self, error: SyntaxError);

    /// Complete tree building. Make sure that
    /// `start_branch` and `finish_branch` calls
    /// are paired!
    fn finish(self) -> Self::Tree;
}

/// Parse a sequence of tokens into the representative node tree
pub(crate) fn parse_with<S: Sink>(
    sink: S,
    text: &str,
    tokens: &[Token],
    parser: fn(&mut Parser),
) -> S::Tree {
    let mut events = {
        let input = input::ParserInput::new(text, tokens);
        let parser_impl = ParserImpl::new(&input);
        let mut parser_api = Parser(parser_impl);
        parser(&mut parser_api);
        parser_api.0.into_events()
    };
    EventProcessor::new(sink, text, tokens, &mut events).process().finish()
}

/// Implementation details of `Parser`, extracted
/// to a separate struct in order not to pollute
/// the public API of the `Parser`.
pub(crate) struct ParserImpl<'t> {
    parser_input: &'t ParserInput<'t>,
    pos: InputPosition,
    events: Vec<Event>,
    steps: Cell<u32>,
}

impl<'t> ParserImpl<'t> {
    pub(crate) fn new(inp: &'t ParserInput<'t>) -> ParserImpl<'t> {
        ParserImpl {
            parser_input: inp,
            pos: InputPosition::new(),
            events: Vec::new(),
            steps: Cell::new(0),
        }
    }

    pub(crate) fn into_events(self) -> Vec<Event> {
        assert_eq!(self.nth(0), EOF);
        self.events
    }

    pub(super) fn next2(&self) -> Option<(SyntaxKind, SyntaxKind)> {
        let c1 = self.parser_input.kind(self.pos);
        let c2 = self.parser_input.kind(self.pos + 1);
        if self.parser_input.token_start_at(self.pos + 1)
            == self.parser_input.token_start_at(self.pos) + self.parser_input.token_len(self.pos)
        {
            Some((c1, c2))
        } else {
            None
        }
    }

    pub(super) fn next3(&self) -> Option<(SyntaxKind, SyntaxKind, SyntaxKind)> {
        let c1 = self.parser_input.kind(self.pos);
        let c2 = self.parser_input.kind(self.pos + 1);
        let c3 = self.parser_input.kind(self.pos + 2);
        if self.parser_input.token_start_at(self.pos + 1)
            == self.parser_input.token_start_at(self.pos) + self.parser_input.token_len(self.pos)
            && self.parser_input.token_start_at(self.pos + 2)
                == self.parser_input.token_start_at(self.pos + 1)
                    + self.parser_input.token_len(self.pos + 1)
        {
            Some((c1, c2, c3))
        } else {
            None
        }
    }

    /// Get the syntax kind of the nth token.
    pub(super) fn nth(&self, n: u32) -> SyntaxKind {
        let steps = self.steps.get();
        assert!(steps <= 10_000_000, "the parser seems stuck");
        self.steps.set(steps + 1);

        self.parser_input.kind(self.pos + n)
    }

    pub(super) fn at_kw(&self, t: &str) -> bool {
        self.parser_input.token_text(self.pos) == t
    }

    /// Start parsing right behind the last event.
    pub(super) fn start(&mut self) -> u32 {
        let pos = self.events.len() as u32;
        self.push_event(Event::tombstone());
        pos
    }

    /// Advances the parser by one token unconditionally.
    pub(super) fn bump(&mut self) {
        let kind = self.nth(0);
        if kind == EOF {
            return;
        }
        self.do_bump(kind, 1);
    }

    pub(super) fn bump_remap(&mut self, kind: SyntaxKind) {
        if self.nth(0) == EOF {
            // TODO: panic!?
            return;
        }
        self.do_bump(kind, 1);
    }

    pub(super) fn bump_compound(&mut self, kind: SyntaxKind, n: u8) {
        self.do_bump(kind, n);
    }

    fn do_bump(&mut self, kind: SyntaxKind, n_raw_tokens: u8) {
        self.pos += u32::from(n_raw_tokens);
        self.push_event(Event::Token { kind, n_raw_tokens });
    }

    /// Append one Error event to the back of events.
    pub(super) fn error(&mut self, msg: String) {
        self.push_event(Event::Error { msg: ParseError(msg) })
    }

    /// Complete an event with appending a `Finish` event.
    pub(super) fn complete(&mut self, pos: u32, kind: SyntaxKind) {
        match self.events[pos as usize] {
            Event::Start { kind: ref mut slot, .. } => {
                *slot = kind;
            }
            _ => unreachable!(),
        }
        self.push_event(Event::Finish);
    }

    /// Ignore the dummy `Start` event.
    pub(super) fn abandon(&mut self, pos: u32) {
        let idx = pos as usize;
        if idx == self.events.len() - 1 {
            match self.events.pop() {
                Some(Event::Start { kind: TOMBSTONE, forward_parent: None }) => (),
                _ => unreachable!(),
            }
        }
    }

    /// Save the relative distance of a completed event to its forward_parent.
    pub(super) fn precede(&mut self, pos: u32) -> u32 {
        let new_pos = self.start();
        match self.events[pos as usize] {
            Event::Start { ref mut forward_parent, .. } => {
                *forward_parent = Some(new_pos - pos);
            }
            _ => unreachable!(),
        }
        new_pos
    }

    fn push_event(&mut self, event: Event) {
        self.events.push(event)
    }
}
