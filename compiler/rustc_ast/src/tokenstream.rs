//! # Token Streams
//!
//! `TokenStream`s represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence (eg stream) of `TokenTree`s,
//! which are themselves a single `Token` or a `Delimited` subsequence of tokens.
//!
//! ## Ownership
//!
//! `TokenStream`s are persistent data structures constructed as ropes with reference
//! counted-children. In general, this means that calling an operation on a `TokenStream`
//! (such as `slice`) produces an entirely new `TokenStream` from the borrowed reference to
//! the original. This essentially coerces `TokenStream`s into 'views' of their subparts,
//! and a borrowed `TokenStream` is sufficient to build an owned `TokenStream` without taking
//! ownership of the original.

use crate::token::{self, DelimToken, Token, TokenKind};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{self, Lrc};
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::{Span, DUMMY_SP};
use smallvec::{smallvec, SmallVec};

use std::{fmt, iter, mem};

/// When the main rust parser encounters a syntax-extension invocation, it
/// parses the arguments to the invocation as a token-tree. This is a very
/// loose structure, such that all sorts of different AST-fragments can
/// be passed to syntax extensions using a uniform type.
///
/// If the syntax extension is an MBE macro, it will attempt to match its
/// LHS token tree against the provided token tree, and if it finds a
/// match, will transcribe the RHS token tree, splicing in any captured
/// `macro_parser::matched_nonterminals` into the `SubstNt`s it finds.
///
/// The RHS of an MBE macro is the only place `SubstNt`s are substituted.
/// Nothing special happens to misnamed or misplaced `SubstNt`s.
#[derive(Debug, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum TokenTree {
    /// A single token
    Token(Token),
    /// A delimited sequence of token trees
    Delimited(DelimSpan, DelimToken, TokenStream),
}

// Ensure all fields of `TokenTree` is `Send` and `Sync`.
#[cfg(parallel_compiler)]
fn _dummy()
where
    Token: Send + Sync,
    DelimSpan: Send + Sync,
    DelimToken: Send + Sync,
    TokenStream: Send + Sync,
{
}

impl TokenTree {
    /// Checks if this TokenTree is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (TokenTree::Token(token), TokenTree::Token(token2)) => token.kind == token2.kind,
            (TokenTree::Delimited(_, delim, tts), TokenTree::Delimited(_, delim2, tts2)) => {
                delim == delim2 && tts.eq_unspanned(&tts2)
            }
            _ => false,
        }
    }

    /// Retrieves the TokenTree's span.
    pub fn span(&self) -> Span {
        match self {
            TokenTree::Token(token) => token.span,
            TokenTree::Delimited(sp, ..) => sp.entire(),
        }
    }

    /// Modify the `TokenTree`'s span in-place.
    pub fn set_span(&mut self, span: Span) {
        match self {
            TokenTree::Token(token) => token.span = span,
            TokenTree::Delimited(dspan, ..) => *dspan = DelimSpan::from_single(span),
        }
    }

    pub fn joint(self) -> TokenStream {
        TokenStream::new(vec![(self, Spacing::Joint)])
    }

    pub fn token(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span))
    }

    /// Returns the opening delimiter as a token tree.
    pub fn open_tt(span: DelimSpan, delim: DelimToken) -> TokenTree {
        TokenTree::token(token::OpenDelim(delim), span.open)
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(span: DelimSpan, delim: DelimToken) -> TokenTree {
        TokenTree::token(token::CloseDelim(delim), span.close)
    }

    pub fn uninterpolate(self) -> TokenTree {
        match self {
            TokenTree::Token(token) => TokenTree::Token(token.uninterpolate().into_owned()),
            tt => tt,
        }
    }
}

impl<CTX> HashStable<CTX> for TokenStream
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        for sub_tt in self.trees() {
            sub_tt.hash_stable(hcx, hasher);
        }
    }
}

pub trait CreateTokenStream: sync::Send + sync::Sync {
    fn create_token_stream(&self) -> TokenStream;
}

impl CreateTokenStream for TokenStream {
    fn create_token_stream(&self) -> TokenStream {
        self.clone()
    }
}

/// A lazy version of `TokenStream`, which defers creation
/// of an actual `TokenStream` until it is needed.
/// `Box` is here only to reduce the structure size.
#[derive(Clone)]
pub struct LazyTokenStream(Lrc<Box<dyn CreateTokenStream>>);

impl LazyTokenStream {
    pub fn new(inner: impl CreateTokenStream + 'static) -> LazyTokenStream {
        LazyTokenStream(Lrc::new(Box::new(inner)))
    }

    pub fn create_token_stream(&self) -> TokenStream {
        self.0.create_token_stream()
    }
}

impl fmt::Debug for LazyTokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt("LazyTokenStream", f)
    }
}

impl<S: Encoder> Encodable<S> for LazyTokenStream {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        // Used by AST json printing.
        Encodable::encode(&self.create_token_stream(), s)
    }
}

impl<D: Decoder> Decodable<D> for LazyTokenStream {
    fn decode(_d: &mut D) -> Result<Self, D::Error> {
        panic!("Attempted to decode LazyTokenStream");
    }
}

impl<CTX> HashStable<CTX> for LazyTokenStream {
    fn hash_stable(&self, _hcx: &mut CTX, _hasher: &mut StableHasher) {
        panic!("Attempted to compute stable hash for LazyTokenStream");
    }
}

/// A `TokenStream` is an abstract sequence of tokens, organized into `TokenTree`s.
///
/// The goal is for procedural macros to work with `TokenStream`s and `TokenTree`s
/// instead of a representation of the abstract syntax tree.
/// Today's `TokenTree`s can still contain AST via `token::Interpolated` for back-compat.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct TokenStream(pub(crate) Lrc<Vec<TreeAndSpacing>>);

pub type TreeAndSpacing = (TokenTree, Spacing);

// `TokenStream` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
rustc_data_structures::static_assert_size!(TokenStream, 8);

#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable)]
pub enum Spacing {
    Alone,
    Joint,
}

impl TokenStream {
    /// Given a `TokenStream` with a `Stream` of only two arguments, return a new `TokenStream`
    /// separating the two arguments with a comma for diagnostic suggestions.
    pub fn add_comma(&self) -> Option<(TokenStream, Span)> {
        // Used to suggest if a user writes `foo!(a b);`
        let mut suggestion = None;
        let mut iter = self.0.iter().enumerate().peekable();
        while let Some((pos, ts)) = iter.next() {
            if let Some((_, next)) = iter.peek() {
                let sp = match (&ts, &next) {
                    (_, (TokenTree::Token(Token { kind: token::Comma, .. }), _)) => continue,
                    (
                        (TokenTree::Token(token_left), Spacing::Alone),
                        (TokenTree::Token(token_right), _),
                    ) if ((token_left.is_ident() && !token_left.is_reserved_ident())
                        || token_left.is_lit())
                        && ((token_right.is_ident() && !token_right.is_reserved_ident())
                            || token_right.is_lit()) =>
                    {
                        token_left.span
                    }
                    ((TokenTree::Delimited(sp, ..), Spacing::Alone), _) => sp.entire(),
                    _ => continue,
                };
                let sp = sp.shrink_to_hi();
                let comma = (TokenTree::token(token::Comma, sp), Spacing::Alone);
                suggestion = Some((pos, comma, sp));
            }
        }
        if let Some((pos, comma, sp)) = suggestion {
            let mut new_stream = Vec::with_capacity(self.0.len() + 1);
            let parts = self.0.split_at(pos + 1);
            new_stream.extend_from_slice(parts.0);
            new_stream.push(comma);
            new_stream.extend_from_slice(parts.1);
            return Some((TokenStream::new(new_stream), sp));
        }
        None
    }
}

impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream::new(vec![(tree, Spacing::Alone)])
    }
}

impl From<TokenTree> for TreeAndSpacing {
    fn from(tree: TokenTree) -> TreeAndSpacing {
        (tree, Spacing::Alone)
    }
}

impl iter::FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(iter: I) -> Self {
        TokenStream::new(iter.into_iter().map(Into::into).collect::<Vec<TreeAndSpacing>>())
    }
}

impl Eq for TokenStream {}

impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.trees().eq(other.trees())
    }
}

impl TokenStream {
    pub fn new(streams: Vec<TreeAndSpacing>) -> TokenStream {
        TokenStream(Lrc::new(streams))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn span(&self) -> Option<Span> {
        match &**self.0 {
            [] => None,
            [(tt, _)] => Some(tt.span()),
            [(tt_start, _), .., (tt_end, _)] => Some(tt_start.span().to(tt_end.span())),
        }
    }

    pub fn from_streams(mut streams: SmallVec<[TokenStream; 2]>) -> TokenStream {
        match streams.len() {
            0 => TokenStream::default(),
            1 => streams.pop().unwrap(),
            _ => {
                // We are going to extend the first stream in `streams` with
                // the elements from the subsequent streams. This requires
                // using `make_mut()` on the first stream, and in practice this
                // doesn't cause cloning 99.9% of the time.
                //
                // One very common use case is when `streams` has two elements,
                // where the first stream has any number of elements within
                // (often 1, but sometimes many more) and the second stream has
                // a single element within.

                // Determine how much the first stream will be extended.
                // Needed to avoid quadratic blow up from on-the-fly
                // reallocations (#57735).
                let num_appends = streams.iter().skip(1).map(|ts| ts.len()).sum();

                // Get the first stream. If it's `None`, create an empty
                // stream.
                let mut iter = streams.drain(..);
                let mut first_stream_lrc = iter.next().unwrap().0;

                // Append the elements to the first stream, after reserving
                // space for them.
                let first_vec_mut = Lrc::make_mut(&mut first_stream_lrc);
                first_vec_mut.reserve(num_appends);
                for stream in iter {
                    first_vec_mut.extend(stream.0.iter().cloned());
                }

                // Create the final `TokenStream`.
                TokenStream(first_stream_lrc)
            }
        }
    }

    pub fn trees_ref(&self) -> CursorRef<'_> {
        CursorRef::new(self)
    }

    pub fn trees(&self) -> Cursor {
        self.clone().into_trees()
    }

    pub fn into_trees(self) -> Cursor {
        Cursor::new(self)
    }

    /// Compares two `TokenStream`s, checking equality without regarding span information.
    pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
        let mut t1 = self.trees();
        let mut t2 = other.trees();
        for (t1, t2) in t1.by_ref().zip(t2.by_ref()) {
            if !t1.eq_unspanned(&t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    pub fn map_enumerated<F: FnMut(usize, &TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        TokenStream(Lrc::new(
            self.0
                .iter()
                .enumerate()
                .map(|(i, (tree, is_joint))| (f(i, tree), *is_joint))
                .collect(),
        ))
    }
}

// 99.5%+ of the time we have 1 or 2 elements in this vector.
#[derive(Clone)]
pub struct TokenStreamBuilder(SmallVec<[TokenStream; 2]>);

impl TokenStreamBuilder {
    pub fn new() -> TokenStreamBuilder {
        TokenStreamBuilder(SmallVec::new())
    }

    pub fn push<T: Into<TokenStream>>(&mut self, stream: T) {
        let mut stream = stream.into();

        // If `self` is not empty and the last tree within the last stream is a
        // token tree marked with `Joint`...
        if let Some(TokenStream(ref mut last_stream_lrc)) = self.0.last_mut() {
            if let Some((TokenTree::Token(last_token), Spacing::Joint)) = last_stream_lrc.last() {
                // ...and `stream` is not empty and the first tree within it is
                // a token tree...
                let TokenStream(ref mut stream_lrc) = stream;
                if let Some((TokenTree::Token(token), spacing)) = stream_lrc.first() {
                    // ...and the two tokens can be glued together...
                    if let Some(glued_tok) = last_token.glue(&token) {
                        // ...then do so, by overwriting the last token
                        // tree in `self` and removing the first token tree
                        // from `stream`. This requires using `make_mut()`
                        // on the last stream in `self` and on `stream`,
                        // and in practice this doesn't cause cloning 99.9%
                        // of the time.

                        // Overwrite the last token tree with the merged
                        // token.
                        let last_vec_mut = Lrc::make_mut(last_stream_lrc);
                        *last_vec_mut.last_mut().unwrap() = (TokenTree::Token(glued_tok), *spacing);

                        // Remove the first token tree from `stream`. (This
                        // is almost always the only tree in `stream`.)
                        let stream_vec_mut = Lrc::make_mut(stream_lrc);
                        stream_vec_mut.remove(0);

                        // Don't push `stream` if it's empty -- that could
                        // block subsequent token gluing, by getting
                        // between two token trees that should be glued
                        // together.
                        if !stream.is_empty() {
                            self.0.push(stream);
                        }
                        return;
                    }
                }
            }
        }
        self.0.push(stream);
    }

    pub fn build(self) -> TokenStream {
        TokenStream::from_streams(self.0)
    }
}

/// By-reference iterator over a `TokenStream`.
#[derive(Clone)]
pub struct CursorRef<'t> {
    stream: &'t TokenStream,
    index: usize,
}

impl<'t> CursorRef<'t> {
    fn new(stream: &TokenStream) -> CursorRef<'_> {
        CursorRef { stream, index: 0 }
    }

    fn next_with_spacing(&mut self) -> Option<&'t TreeAndSpacing> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree
        })
    }
}

impl<'t> Iterator for CursorRef<'t> {
    type Item = &'t TokenTree;

    fn next(&mut self) -> Option<&'t TokenTree> {
        self.next_with_spacing().map(|(tree, _)| tree)
    }
}

/// Owning by-value iterator over a `TokenStream`.
/// FIXME: Many uses of this can be replaced with by-reference iterator to avoid clones.
#[derive(Clone)]
pub struct Cursor {
    pub stream: TokenStream,
    index: usize,
}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.next_with_spacing().map(|(tree, _)| tree)
    }
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Cursor { stream, index: 0 }
    }

    pub fn next_with_spacing(&mut self) -> Option<TreeAndSpacing> {
        if self.index < self.stream.len() {
            self.index += 1;
            Some(self.stream.0[self.index - 1].clone())
        } else {
            None
        }
    }

    pub fn append(&mut self, new_stream: TokenStream) {
        if new_stream.is_empty() {
            return;
        }
        let index = self.index;
        let stream = mem::take(&mut self.stream);
        *self = TokenStream::from_streams(smallvec![stream, new_stream]).into_trees();
        self.index = index;
    }

    pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
        self.stream.0[self.index..].get(n).map(|(tree, _)| tree)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub struct DelimSpan {
    pub open: Span,
    pub close: Span,
}

impl DelimSpan {
    pub fn from_single(sp: Span) -> Self {
        DelimSpan { open: sp, close: sp }
    }

    pub fn from_pair(open: Span, close: Span) -> Self {
        DelimSpan { open, close }
    }

    pub fn dummy() -> Self {
        Self::from_single(DUMMY_SP)
    }

    pub fn entire(self) -> Span {
        self.open.with_hi(self.close.hi())
    }
}
