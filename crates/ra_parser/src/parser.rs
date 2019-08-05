use std::cell::Cell;

use drop_bomb::DropBomb;

use crate::{
    event::Event,
    ParseError,
    SyntaxKind::{self, EOF, ERROR, TOMBSTONE},
    Token, TokenSet, TokenSource, T,
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
    token_source: &'t mut dyn TokenSource,
    events: Vec<Event>,
    steps: Cell<u32>,
}

impl<'t> Parser<'t> {
    pub(super) fn new(token_source: &'t mut dyn TokenSource) -> Parser<'t> {
        Parser { token_source, events: Vec::new(), steps: Cell::new(0) }
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
        let c1 = self.nth(0);
        let c2 = self.nth(1);

        if self.token_source.current().is_jointed_to_next {
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
        let c1 = self.nth(0);
        let c2 = self.nth(1);
        let c3 = self.nth(2);
        if self.token_source.current().is_jointed_to_next
            && self.token_source.lookahead_nth(1).is_jointed_to_next
        {
            Some((c1, c2, c3))
        } else {
            None
        }
    }

    /// Lookahead operation: returns the kind of the next nth
    /// token.
    pub(crate) fn nth(&self, n: usize) -> SyntaxKind {
        assert!(n <= 3);

        let steps = self.steps.get();
        assert!(steps <= 10_000_000, "the parser seems stuck");
        self.steps.set(steps + 1);

        // It is beecause the Dollar will appear between nth
        // Following code skips through it
        let mut non_dollars_count = 0;
        let mut i = 0;

        loop {
            let token = self.token_source.lookahead_nth(i);
            let mut kind = token.kind;
            if let Some((composited, step)) = self.is_composite(token, i) {
                kind = composited;
                i += step;
            } else {
                i += 1;
            }

            match kind {
                EOF => return EOF,
                SyntaxKind::L_DOLLAR | SyntaxKind::R_DOLLAR => {}
                _ if non_dollars_count == n => return kind,
                _ => non_dollars_count += 1,
            }
        }
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
        self.token_source.is_keyword(kw)
    }

    /// Starts a new node in the syntax tree. All nodes and tokens
    /// consumed between the `start` and the corresponding `Marker::complete`
    /// belong to the same node.
    pub(crate) fn start(&mut self) -> Marker {
        let pos = self.events.len() as u32;
        self.push_event(Event::tombstone());
        Marker::new(pos)
    }

    /// Advances the parser by one token unconditionally
    /// Mainly use in `token_tree` parsing
    pub(crate) fn bump_raw(&mut self) {
        let mut kind = self.token_source.current().kind;

        // Skip dollars, do_bump will eat these later
        let mut i = 0;
        while kind == SyntaxKind::L_DOLLAR || kind == SyntaxKind::R_DOLLAR {
            kind = self.token_source.lookahead_nth(i).kind;
            i += 1;
        }

        if kind == EOF {
            return;
        }
        self.do_bump(kind, 1);
    }

    /// Advances the parser by one token with composite puncts handled
    pub(crate) fn bump(&mut self) {
        let kind = self.nth(0);
        if kind == EOF {
            return;
        }

        use SyntaxKind::*;

        // Handle parser composites
        match kind {
            T![...] | T![..=] => {
                self.bump_compound(kind, 3);
            }
            T![..] | T![::] | T![==] | T![=>] | T![!=] | T![->] => {
                self.bump_compound(kind, 2);
            }
            _ => {
                self.do_bump(kind, 1);
            }
        }
    }

    /// Advances the parser by one token, remapping its kind.
    /// This is useful to create contextual keywords from
    /// identifiers. For example, the lexer creates an `union`
    /// *identifier* token, but the parser remaps it to the
    /// `union` keyword, and keyword is what ends up in the
    /// final tree.
    pub(crate) fn bump_remap(&mut self, kind: SyntaxKind) {
        if self.nth(0) == EOF {
            // FIXME: panic!?
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
    /// FIXME: this should be much more fancy and support
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
        if self.at(T!['{']) || self.at(T!['}']) || self.at_ts(recovery) {
            self.error(message);
        } else {
            let m = self.start();
            self.error(message);
            self.bump();
            m.complete(self, ERROR);
        };
    }

    fn do_bump(&mut self, kind: SyntaxKind, n_raw_tokens: u8) {
        self.eat_dollars();

        for _ in 0..n_raw_tokens {
            self.token_source.bump();
        }

        self.push_event(Event::Token { kind, n_raw_tokens });
    }

    fn push_event(&mut self, event: Event) {
        self.events.push(event)
    }

    /// helper function for check if it is composite.
    fn is_composite(&self, first: Token, n: usize) -> Option<(SyntaxKind, usize)> {
        // We assume the dollars will not occuried between
        // mult-byte tokens

        let jn1 = first.is_jointed_to_next;
        if !jn1 && first.kind != T![-] {
            return None;
        }

        let second = self.token_source.lookahead_nth(n + 1);
        if first.kind == T![-] && second.kind == T![>] {
            return Some((T![->], 2));
        }
        if !jn1 {
            return None;
        }

        match (first.kind, second.kind) {
            (T![:], T![:]) => return Some((T![::], 2)),
            (T![=], T![=]) => return Some((T![==], 2)),
            (T![=], T![>]) => return Some((T![=>], 2)),
            (T![!], T![=]) => return Some((T![!=], 2)),
            _ => {}
        }

        if first.kind != T![.] || second.kind != T![.] {
            return None;
        }

        let third = self.token_source.lookahead_nth(n + 2);

        let jn2 = second.is_jointed_to_next;
        let la3 = third.kind;

        if jn2 && la3 == T![.] {
            return Some((T![...], 3));
        }
        if la3 == T![=] {
            return Some((T![..=], 3));
        }
        return Some((T![..], 2));
    }

    fn eat_dollars(&mut self) {
        loop {
            match self.token_source.current().kind {
                k @ SyntaxKind::L_DOLLAR | k @ SyntaxKind::R_DOLLAR => {
                    self.token_source.bump();
                    self.push_event(Event::Token { kind: k, n_raw_tokens: 1 });
                }
                _ => {
                    return;
                }
            }
        }
    }

    pub(crate) fn eat_l_dollars(&mut self) -> usize {
        let mut ate_count = 0;
        loop {
            match self.token_source.current().kind {
                k @ SyntaxKind::L_DOLLAR => {
                    self.token_source.bump();
                    self.push_event(Event::Token { kind: k, n_raw_tokens: 1 });
                    ate_count += 1;
                }
                _ => {
                    return ate_count;
                }
            }
        }
    }

    pub(crate) fn eat_r_dollars(&mut self, max_count: usize) -> usize {
        let mut ate_count = 0;
        loop {
            match self.token_source.current().kind {
                k @ SyntaxKind::R_DOLLAR => {
                    self.token_source.bump();
                    self.push_event(Event::Token { kind: k, n_raw_tokens: 1 });
                    ate_count += 1;

                    if max_count >= ate_count {
                        return ate_count;
                    }
                }
                _ => {
                    return ate_count;
                }
            }
        }
    }

    pub(crate) fn at_l_dollar(&self) -> bool {
        let kind = self.token_source.current().kind;
        (kind == SyntaxKind::L_DOLLAR)
    }

    pub(crate) fn at_r_dollar(&self) -> bool {
        let kind = self.token_source.current().kind;
        (kind == SyntaxKind::R_DOLLAR)
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
        let finish_pos = p.events.len() as u32;
        p.push_event(Event::Finish);
        CompletedMarker::new(self.pos, finish_pos, kind)
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

pub(crate) struct CompletedMarker {
    start_pos: u32,
    finish_pos: u32,
    kind: SyntaxKind,
}

impl CompletedMarker {
    fn new(start_pos: u32, finish_pos: u32, kind: SyntaxKind) -> Self {
        CompletedMarker { start_pos, finish_pos, kind }
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
        let idx = self.start_pos as usize;
        match p.events[idx] {
            Event::Start { ref mut forward_parent, .. } => {
                *forward_parent = Some(new_pos.pos - self.start_pos);
            }
            _ => unreachable!(),
        }
        new_pos
    }

    /// Undo this completion and turns into a `Marker`
    pub(crate) fn undo_completion(self, p: &mut Parser) -> Marker {
        let start_idx = self.start_pos as usize;
        let finish_idx = self.finish_pos as usize;
        match p.events[start_idx] {
            Event::Start { ref mut kind, forward_parent: None } => *kind = TOMBSTONE,
            _ => unreachable!(),
        }
        match p.events[finish_idx] {
            ref mut slot @ Event::Finish => *slot = Event::tombstone(),
            _ => unreachable!(),
        }
        Marker::new(self.start_pos)
    }

    pub(crate) fn kind(&self) -> SyntaxKind {
        self.kind
    }
}
