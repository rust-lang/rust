use {SyntaxKind, TextUnit, Token};
use super::Event;
use super::super::is_insignificant;
use SyntaxKind::{EOF, TOMBSTONE};

pub(crate) struct Marker {
    pos: u32,
}

impl Marker {
    pub fn complete(self, p: &mut Parser, kind: SyntaxKind) -> CompleteMarker {
        match self.event(p) {
            &mut Event::Start {
                kind: ref mut slot, ..
            } => {
                *slot = kind;
            }
            _ => unreachable!(),
        }
        p.event(Event::Finish);
        let result = CompleteMarker { pos: self.pos };
        ::std::mem::forget(self);
        result
    }

    pub fn abandon(self, p: &mut Parser) {
        let idx = self.pos as usize;
        if idx == p.events.len() - 1 {
            match p.events.pop() {
                Some(Event::Start {
                    kind: TOMBSTONE,
                    forward_parent: None,
                }) => (),
                _ => unreachable!(),
            }
        }
        ::std::mem::forget(self);
    }

    fn event<'p>(&self, p: &'p mut Parser) -> &'p mut Event {
        &mut p.events[self.idx()]
    }

    fn idx(&self) -> usize {
        self.pos as usize
    }
}

impl Drop for Marker {
    fn drop(&mut self) {
        if !::std::thread::panicking() {
            panic!("Each marker should be eithe completed or abandoned");
        }
    }
}

pub(crate) struct CompleteMarker {
    pos: u32,
}

impl CompleteMarker {
    pub(crate) fn precede(self, p: &mut Parser) -> Marker {
        let m = p.start();
        match p.events[self.pos as usize] {
            Event::Start {
                ref mut forward_parent,
                ..
            } => {
                *forward_parent = Some(m.pos - self.pos);
            }
            _ => unreachable!(),
        }
        m
    }
}

pub(crate) struct TokenSet {
    pub tokens: &'static [SyntaxKind],
}

impl TokenSet {
    pub fn contains(&self, kind: SyntaxKind) -> bool {
        self.tokens.contains(&kind)
    }
}

#[macro_export]
macro_rules! token_set {
    ($($t:ident),*) => {
        TokenSet {
            tokens: &[$($t),*],
        }
    };

    ($($t:ident),* ,) => {
        token_set!($($t),*)
    };
}

pub(crate) struct Parser<'t> {
    #[allow(unused)]
    text: &'t str,
    #[allow(unused)]
    start_offsets: Vec<TextUnit>,
    tokens: Vec<Token>, // non-whitespace tokens

    pos: usize,
    events: Vec<Event>,
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
        }
    }

    pub(crate) fn into_events(self) -> Vec<Event> {
        assert_eq!(self.current(), EOF);
        self.events
    }

    pub(crate) fn current(&self) -> SyntaxKind {
        if self.pos == self.tokens.len() {
            return EOF;
        }
        self.tokens[self.pos].kind
    }

    pub(crate) fn start(&mut self) -> Marker {
        let m = Marker {
            pos: self.events.len() as u32,
        };
        self.event(Event::Start {
            kind: TOMBSTONE,
            forward_parent: None,
        });
        m
    }

    pub(crate) fn error<'p>(&'p mut self) -> ErrorBuilder<'p, 't> {
        ErrorBuilder::new(self)
    }

    pub(crate) fn bump(&mut self) -> SyntaxKind {
        let kind = self.current();
        if kind == EOF {
            return EOF;
        }
        self.pos += 1;
        self.event(Event::Token {
            kind,
            n_raw_tokens: 1,
        });
        kind
    }

    pub(crate) fn raw_lookahead(&self, n: usize) -> SyntaxKind {
        self.tokens.get(self.pos + n).map(|t| t.kind).unwrap_or(EOF)
    }

    fn event(&mut self, event: Event) {
        self.events.push(event)
    }
}

pub(crate) struct ErrorBuilder<'p, 't: 'p> {
    message: Option<String>,
    parser: &'p mut Parser<'t>,
}

impl<'t, 'p> ErrorBuilder<'p, 't> {
    fn new(parser: &'p mut Parser<'t>) -> Self {
        ErrorBuilder {
            message: None,
            parser,
        }
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
