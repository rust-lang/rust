use std::cell::Cell;

use drop_bomb::DropBomb;

use crate::{
    SyntaxKind::{self, ERROR, EOF, TOMBSTONE},
    TokenSource, ParseError, TokenSet,
    event::Event,
};

/// `Parser` struct provides the low-level API for
/// navigating through the stream of tokens and
/// constructing the parse tree. The actual parsing
/// happens in the `grammar` module.
///
/// However, the result of this `Parser` is not a real
/// tree, but rather a flat stream of events of the form
/// "start expression, consume number literal,
/// finish expression". See `Event` docs for more.
pub(crate) struct Parser<'t> {
    token_source: &'t dyn TokenSource,
    token_pos: usize,
    events: Vec<Event>,
    steps: Cell<u32>,
}

impl<'t> Parser<'t> {
    pub(super) fn new(token_source: &'t dyn TokenSource) -> Parser<'t> {
        Parser { token_source, token_pos: 0, events: Vec::new(), steps: Cell::new(0) }
    }

    pub(crate) fn finish(self) -> Vec<Event> {
        self.events
    }

    /// Returns the kind of the current token.
    /// If parser has already reached the end of input,
    /// the special `EOF` kind is returned.
    pub(crate) fn current(&self) -> SyntaxKind {
        self.nth(0)
    }

    /// Returns the kinds of the current two tokens, if they are not separated
    /// by trivia.
    ///
    /// Useful for parsing things like `>>`.
    pub(crate) fn current2(&self) -> Option<(SyntaxKind, SyntaxKind)> {
        let c1 = self.token_source.token_kind(self.token_pos);
        let c2 = self.token_source.token_kind(self.token_pos + 1);
        if self.token_source.is_token_joint_to_next(self.token_pos) {
            Some((c1, c2))
        } else {
            None
        }
    }

    /// Returns the kinds of the current three tokens, if they are not separated
    /// by trivia.
    ///
    /// Useful for parsing things like `=>>`.
    pub(crate) fn current3(&self) -> Option<(SyntaxKind, SyntaxKind, SyntaxKind)> {
        let c1 = self.token_source.token_kind(self.token_pos);
        let c2 = self.token_source.token_kind(self.token_pos + 1);
        let c3 = self.token_source.token_kind(self.token_pos + 2);
        if self.token_source.is_token_joint_to_next(self.token_pos)
            && self.token_source.is_token_joint_to_next(self.token_pos + 1)
        {
            Some((c1, c2, c3))
        } else {
            None
        }
    }

    /// Lookahead operation: returns the kind of the next nth
    /// token.
    pub(crate) fn nth(&self, n: usize) -> SyntaxKind {
        let steps = self.steps.get();
        assert!(steps <= 10_000_000, "the parser seems stuck");
        self.steps.set(steps + 1);
        self.token_source.token_kind(self.token_pos + n)
    }

    /// Checks if the current token is `kind`.
    pub(crate) fn at(&self, kind: SyntaxKind) -> bool {
        self.current() == kind
    }

    /// Checks if the current token is in `kinds`.
    pub(crate) fn at_ts(&self, kinds: TokenSet) -> bool {
        kinds.contains(self.current())
    }

    /// Checks if the current token is contextual keyword with text `t`.
    pub(crate) fn at_contextual_kw(&self, kw: &str) -> bool {
        self.token_source.is_keyword(self.token_pos, kw)
    }

    /// Starts a new node in the syntax tree. All nodes and tokens
    /// consumed between the `start` and the corresponding `Marker::complete`
    /// belong to the same node.
    pub(crate) fn start(&mut self) -> Marker {
        let pos = self.events.len() as u32;
        self.push_event(Event::tombstone());
        Marker::new(pos)
    }

    /// Advances the parser by one token unconditionally.
    pub(crate) fn bump(&mut self) {
        let kind = self.nth(0);
        if kind == EOF {
            return;
        }
        self.do_bump(kind, 1);
    }

    /// Advances the parser by one token, remapping its kind.
    /// This is useful to create contextual keywords from
    /// identifiers. For example, the lexer creates an `union`
    /// *identifier* token, but the parser remaps it to the
    /// `union` keyword, and keyword is what ends up in the
    /// final tree.
    pub(crate) fn bump_remap(&mut self, kind: SyntaxKind) {
        if self.nth(0) == EOF {
            // TODO: panic!?
            return;
        }
        self.do_bump(kind, 1);
    }

    /// Advances the parser by `n` tokens, remapping its kind.
    /// This is useful to create compound tokens from parts. For
    /// example, an `<<` token is two consecutive remapped `<` tokens
    pub(crate) fn bump_compound(&mut self, kind: SyntaxKind, n: u8) {
        self.do_bump(kind, n);
    }

    /// Emit error with the `message`
    /// TODO: this should be much more fancy and support
    /// structured errors with spans and notes, like rustc
    /// does.
    pub(crate) fn error<T: Into<String>>(&mut self, message: T) {
        let msg = ParseError(message.into());
        self.push_event(Event::Error { msg })
    }

    /// Consume the next token if `kind` matches.
    pub(crate) fn eat(&mut self, kind: SyntaxKind) -> bool {
        if !self.at(kind) {
            return false;
        }
        self.bump();
        true
    }

    /// Consume the next token if it is `kind` or emit an error
    /// otherwise.
    pub(crate) fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.eat(kind) {
            return true;
        }
        self.error(format!("expected {:?}", kind));
        false
    }

    /// Create an error node and consume the next token.
    pub(crate) fn err_and_bump(&mut self, message: &str) {
        self.err_recover(message, TokenSet::empty());
    }

    /// Create an error node and consume the next token.
    pub(crate) fn err_recover(&mut self, message: &str, recovery: TokenSet) {
        if self.at(SyntaxKind::L_CURLY) || self.at(SyntaxKind::R_CURLY) || self.at_ts(recovery) {
            self.error(message);
        } else {
            let m = self.start();
            self.error(message);
            self.bump();
            m.complete(self, ERROR);
        };
    }

    fn do_bump(&mut self, kind: SyntaxKind, n_raw_tokens: u8) {
        self.token_pos += usize::from(n_raw_tokens);
        self.push_event(Event::Token { kind, n_raw_tokens });
    }

    fn push_event(&mut self, event: Event) {
        self.events.push(event)
    }
}

/// See `Parser::start`.
pub(crate) struct Marker {
    pos: u32,
    bomb: DropBomb,
}

impl Marker {
    fn new(pos: u32) -> Marker {
        Marker { pos, bomb: DropBomb::new("Marker must be either completed or abandoned") }
    }

    /// Finishes the syntax tree node and assigns `kind` to it,
    /// and mark the create a `CompletedMarker` for possible future
    /// operation like `.precede()` to deal with forward_parent.
    pub(crate) fn complete(mut self, p: &mut Parser, kind: SyntaxKind) -> CompletedMarker {
        self.bomb.defuse();
        let idx = self.pos as usize;
        match p.events[idx] {
            Event::Start { kind: ref mut slot, .. } => {
                *slot = kind;
            }
            _ => unreachable!(),
        }
        p.push_event(Event::Finish);
        CompletedMarker::new(self.pos, kind)
    }

    /// Abandons the syntax tree node. All its children
    /// are attached to its parent instead.
    pub(crate) fn abandon(mut self, p: &mut Parser) {
        self.bomb.defuse();
        let idx = self.pos as usize;
        if idx == p.events.len() - 1 {
            match p.events.pop() {
                Some(Event::Start { kind: TOMBSTONE, forward_parent: None }) => (),
                _ => unreachable!(),
            }
        }
    }
}

pub(crate) struct CompletedMarker(u32, SyntaxKind);

impl CompletedMarker {
    fn new(pos: u32, kind: SyntaxKind) -> Self {
        CompletedMarker(pos, kind)
    }

    /// This method allows to create a new node which starts
    /// *before* the current one. That is, parser could start
    /// node `A`, then complete it, and then after parsing the
    /// whole `A`, decide that it should have started some node
    /// `B` before starting `A`. `precede` allows to do exactly
    /// that. See also docs about `forward_parent` in `Event::Start`.
    ///
    /// Given completed events `[START, FINISH]` and its corresponding
    /// `CompletedMarker(pos: 0, _)`.
    /// Append a new `START` events as `[START, FINISH, NEWSTART]`,
    /// then mark `NEWSTART` as `START`'s parent with saving its relative
    /// distance to `NEWSTART` into forward_parent(=2 in this case);
    pub(crate) fn precede(self, p: &mut Parser) -> Marker {
        let new_pos = p.start();
        let idx = self.0 as usize;
        match p.events[idx] {
            Event::Start { ref mut forward_parent, .. } => {
                *forward_parent = Some(new_pos.pos - self.0);
            }
            _ => unreachable!(),
        }
        new_pos
    }

    pub(crate) fn kind(&self) -> SyntaxKind {
        self.1
    }
}
