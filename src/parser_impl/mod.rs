mod event;
mod input;

use {
    grammar,
    lexer::Token,
    parser_api::Parser,
    parser_impl::{
        event::{process, Event},
        input::{InputPosition, ParserInput},
    },
    TextUnit,
};

use SyntaxKind::{self, EOF, TOMBSTONE};

pub(crate) trait Sink {
    type Tree;

    fn new(text: String) -> Self;

    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit);
    fn start_internal(&mut self, kind: SyntaxKind);
    fn finish_internal(&mut self);
    fn error(&mut self, err: String);
    fn finish(self) -> Self::Tree;
}

/// Parse a sequence of tokens into the representative node tree
pub(crate) fn parse<S: Sink>(text: String, tokens: &[Token]) -> S::Tree {
    let events = {
        let input = input::ParserInput::new(&text, tokens);
        let parser_impl = ParserImpl::new(&input);
        let mut parser_api = Parser(parser_impl);
        grammar::file(&mut parser_api);
        parser_api.0.into_events()
    };
    let mut sink = S::new(text);
    process(&mut sink, tokens, events);
    sink.finish()
}

/// Implementation details of `Parser`, extracted
/// to a separate struct in order not to pollute
/// the public API of the `Parser`.
pub(crate) struct ParserImpl<'t> {
    inp: &'t ParserInput<'t>,

    pos: InputPosition,
    events: Vec<Event>,
}

impl<'t> ParserImpl<'t> {
    pub(crate) fn new(inp: &'t ParserInput<'t>) -> ParserImpl<'t> {
        ParserImpl {
            inp,

            pos: InputPosition::new(),
            events: Vec::new(),
        }
    }

    pub(crate) fn into_events(self) -> Vec<Event> {
        assert_eq!(self.nth(0), EOF);
        self.events
    }

    pub(super) fn nth(&self, n: u32) -> SyntaxKind {
        self.inp.kind(self.pos + n)
    }

    pub(super) fn at_kw(&self, t: &str) -> bool {
        self.inp.text(self.pos) == t
    }

    pub(super) fn start(&mut self) -> u32 {
        let pos = self.events.len() as u32;
        self.event(Event::Start {
            kind: TOMBSTONE,
            forward_parent: None,
        });
        pos
    }

    pub(super) fn bump(&mut self) {
        let kind = self.nth(0);
        if kind == EOF {
            return;
        }
        self.do_bump(kind);
    }

    pub(super) fn bump_remap(&mut self, kind: SyntaxKind) {
        if self.nth(0) == EOF {
            // TODO: panic!?
            return;
        }
        self.do_bump(kind);
    }

    fn do_bump(&mut self, kind: SyntaxKind) {
        self.pos += 1;
        self.event(Event::Token {
            kind,
            n_raw_tokens: 1,
        });
    }

    pub(super) fn error(&mut self, msg: String) {
        self.event(Event::Error { msg })
    }

    pub(super) fn complete(&mut self, pos: u32, kind: SyntaxKind) {
        match self.events[pos as usize] {
            Event::Start {
                kind: ref mut slot, ..
            } => {
                *slot = kind;
            }
            _ => unreachable!(),
        }
        self.event(Event::Finish);
    }

    pub(super) fn abandon(&mut self, pos: u32) {
        let idx = pos as usize;
        if idx == self.events.len() - 1 {
            match self.events.pop() {
                Some(Event::Start {
                    kind: TOMBSTONE,
                    forward_parent: None,
                }) => (),
                _ => unreachable!(),
            }
        }
    }

    pub(super) fn precede(&mut self, pos: u32) -> u32 {
        let new_pos = self.start();
        match self.events[pos as usize] {
            Event::Start {
                ref mut forward_parent,
                ..
            } => {
                *forward_parent = Some(new_pos - pos);
            }
            _ => unreachable!(),
        }
        new_pos
    }

    fn event(&mut self, event: Event) {
        self.events.push(event)
    }
}
