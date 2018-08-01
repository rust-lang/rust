use {
    parser_impl::ParserImpl,
    SyntaxKind::{self, ERROR},
};

#[derive(Clone, Copy)]
pub(crate) struct TokenSet(pub(crate) u128);
fn mask(kind: SyntaxKind) -> u128 {
    1u128 << (kind as usize)
}

impl TokenSet {
    pub fn contains(&self, kind: SyntaxKind) -> bool {
        self.0 & mask(kind) != 0
    }
}

#[macro_export]
macro_rules! token_set {
    ($($t:ident),*) => {
        TokenSet($(1u128 << ($t as usize))|*)
    };

    ($($t:ident),* ,) => {
        token_set!($($t),*)
    };
}

#[macro_export]
macro_rules! token_set_union {
    ($($ts:expr),*) => {
        TokenSet($($ts.0)|*)
    };

    ($($ts:expr),* ,) => {
        token_set_union!($($ts),*)
    };
}

/// `Parser` struct provides the low-level API for
/// navigating through the stream of tokens and
/// constructing the parse tree. The actual parsing
/// happens in the `grammar` module.
///
/// However, the result of this `Parser` is not a real
/// tree, but rather a flat stream of events of the form
/// "start expression, consume number literal,
/// finish expression". See `Event` docs for more.
pub(crate) struct Parser<'t>(pub(super) ParserImpl<'t>);

impl<'t> Parser<'t> {
    /// Returns the kind of the current token.
    /// If parser has already reached the end of input,
    /// the special `EOF` kind is returned.
    pub(crate) fn current(&self) -> SyntaxKind {
        self.nth(0)
    }

    /// Lookahead operation: returns the kind of the next nth
    /// token.
    pub(crate) fn nth(&self, n: u32) -> SyntaxKind {
        self.0.nth(n)
    }

    /// Checks if the current token is `kind`.
    pub(crate) fn at(&self, kind: SyntaxKind) -> bool {
        self.current() == kind
    }

    /// Checks if the current token is contextual keyword with text `t`.
    pub(crate) fn at_contextual_kw(&self, t: &str) -> bool {
        self.0.at_kw(t)
    }

    /// Starts a new node in the syntax tree. All nodes and tokens
    /// consumed between the `start` and the corresponding `Marker::complete`
    /// belong to the same node.
    pub(crate) fn start(&mut self) -> Marker {
        Marker(self.0.start())
    }

    /// Advances the parser by one token.
    pub(crate) fn bump(&mut self) {
        self.0.bump();
    }

    /// Advances the parser by one token, remapping its kind.
    /// This is useful to create contextual keywords from
    /// identifiers. For example, the lexer creates an `union`
    /// *identifier* token, but the parser remaps it to the
    /// `union` keyword, and keyword is what ends up in the
    /// final tree.
    pub(crate) fn bump_remap(&mut self, kind: SyntaxKind) {
        self.0.bump_remap(kind);
    }

    /// Emit error with the `message`
    /// TODO: this should be much more fancy and support
    /// structured errors with spans and notes, like rustc
    /// does.
    pub(crate) fn error<T: Into<String>>(&mut self, message: T) {
        self.0.error(message.into())
    }

    /// Consume the next token if it is `kind`.
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
        let m = self.start();
        self.error(message);
        self.bump();
        m.complete(self, ERROR);
    }
}

/// See `Parser::start`.
pub(crate) struct Marker(u32);

impl Marker {
    /// Finishes the syntax tree node and assigns `kind` to it.
    pub(crate) fn complete(self, p: &mut Parser, kind: SyntaxKind) -> CompletedMarker {
        let pos = self.0;
        ::std::mem::forget(self);
        p.0.complete(pos, kind);
        CompletedMarker(pos)
    }

    /// Abandons the syntax tree node. All its children
    /// are attached to its parent instead.
    pub(crate) fn abandon(self, p: &mut Parser) {
        let pos = self.0;
        ::std::mem::forget(self);
        p.0.abandon(pos);
    }
}

impl Drop for Marker {
    fn drop(&mut self) {
        if !::std::thread::panicking() {
            panic!("Marker must be either completed or abandoned");
        }
    }
}

pub(crate) struct CompletedMarker(u32);

impl CompletedMarker {
    /// This one is tricky :-)
    /// This method allows to create a new node which starts
    /// *before* the current one. That is, parser could start
    /// node `A`, then complete it, and then after parsing the
    /// whole `A`, decide that it should have started some node
    /// `B` before starting `A`. `precede` allows to do exactly
    /// that. See also docs about `forward_parent` in `Event::Start`.
    pub(crate) fn precede(self, p: &mut Parser) -> Marker {
        Marker(p.0.precede(self.0))
    }
}
