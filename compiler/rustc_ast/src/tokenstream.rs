//! # Token Streams
//!
//! `TokenStream`s represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence of [`TokenTree`]s,
//! which are themselves a single [`Token`] or a `Delimited` subsequence of tokens.
//!
//! ## Ownership
//!
//! `TokenStream`s are persistent data structures constructed as ropes with reference
//! counted-children. In general, this means that calling an operation on a `TokenStream`
//! (such as `slice`) produces an entirely new `TokenStream` from the borrowed reference to
//! the original. This essentially coerces `TokenStream`s into "views" of their subparts,
//! and a borrowed `TokenStream` is sufficient to build an owned `TokenStream` without taking
//! ownership of the original.

use crate::ast::StmtKind;
use crate::ast_traits::{HasAttrs, HasSpan, HasTokens};
use crate::token::{self, Delimiter, Nonterminal, Token, TokenKind};
use crate::AttrVec;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync::{self, Lrc};
use rustc_macros::HashStable_Generic;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_span::{Span, DUMMY_SP};
use smallvec::{smallvec, SmallVec};

use std::{fmt, iter};

/// When the main Rust parser encounters a syntax-extension invocation, it
/// parses the arguments to the invocation as a token tree. This is a very
/// loose structure, such that all sorts of different AST fragments can
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
    /// A single token.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    Delimited(DelimSpan, Delimiter, TokenStream),
}

// This type is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(TokenTree, 32);

// Ensure all fields of `TokenTree` is `Send` and `Sync`.
#[cfg(parallel_compiler)]
fn _dummy()
where
    Token: Send + Sync,
    DelimSpan: Send + Sync,
    Delimiter: Send + Sync,
    TokenStream: Send + Sync,
{
}

impl TokenTree {
    /// Checks if this `TokenTree` is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (TokenTree::Token(token, _), TokenTree::Token(token2, _)) => token.kind == token2.kind,
            (TokenTree::Delimited(_, delim, tts), TokenTree::Delimited(_, delim2, tts2)) => {
                delim == delim2 && tts.eq_unspanned(&tts2)
            }
            _ => false,
        }
    }

    /// Retrieves the `TokenTree`'s span.
    pub fn span(&self) -> Span {
        match self {
            TokenTree::Token(token, _) => token.span,
            TokenTree::Delimited(sp, ..) => sp.entire(),
        }
    }

    /// Modify the `TokenTree`'s span in-place.
    pub fn set_span(&mut self, span: Span) {
        match self {
            TokenTree::Token(token, _) => token.span = span,
            TokenTree::Delimited(dspan, ..) => *dspan = DelimSpan::from_single(span),
        }
    }

    // Create a `TokenTree::Token` with alone spacing.
    pub fn token_alone(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span), Spacing::Alone)
    }

    // Create a `TokenTree::Token` with joint spacing.
    pub fn token_joint(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span), Spacing::Joint)
    }

    pub fn uninterpolate(self) -> TokenTree {
        match self {
            TokenTree::Token(token, spacing) => {
                TokenTree::Token(token.uninterpolate().into_owned(), spacing)
            }
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
    fn create_token_stream(&self) -> AttrAnnotatedTokenStream;
}

impl CreateTokenStream for AttrAnnotatedTokenStream {
    fn create_token_stream(&self) -> AttrAnnotatedTokenStream {
        self.clone()
    }
}

/// A lazy version of [`TokenStream`], which defers creation
/// of an actual `TokenStream` until it is needed.
/// `Box` is here only to reduce the structure size.
#[derive(Clone)]
pub struct LazyTokenStream(Lrc<Box<dyn CreateTokenStream>>);

impl LazyTokenStream {
    pub fn new(inner: impl CreateTokenStream + 'static) -> LazyTokenStream {
        LazyTokenStream(Lrc::new(Box::new(inner)))
    }

    pub fn create_token_stream(&self) -> AttrAnnotatedTokenStream {
        self.0.create_token_stream()
    }
}

impl fmt::Debug for LazyTokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyTokenStream({:?})", self.create_token_stream())
    }
}

impl<S: Encoder> Encodable<S> for LazyTokenStream {
    fn encode(&self, s: &mut S) {
        // Used by AST json printing.
        Encodable::encode(&self.create_token_stream(), s);
    }
}

impl<D: Decoder> Decodable<D> for LazyTokenStream {
    fn decode(_d: &mut D) -> Self {
        panic!("Attempted to decode LazyTokenStream");
    }
}

impl<CTX> HashStable<CTX> for LazyTokenStream {
    fn hash_stable(&self, _hcx: &mut CTX, _hasher: &mut StableHasher) {
        panic!("Attempted to compute stable hash for LazyTokenStream");
    }
}

/// A `AttrAnnotatedTokenStream` is similar to a `TokenStream`, but with extra
/// information about the tokens for attribute targets. This is used
/// during expansion to perform early cfg-expansion, and to process attributes
/// during proc-macro invocations.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct AttrAnnotatedTokenStream(pub Lrc<Vec<(AttrAnnotatedTokenTree, Spacing)>>);

/// Like `TokenTree`, but for `AttrAnnotatedTokenStream`
#[derive(Clone, Debug, Encodable, Decodable)]
pub enum AttrAnnotatedTokenTree {
    Token(Token),
    Delimited(DelimSpan, Delimiter, AttrAnnotatedTokenStream),
    /// Stores the attributes for an attribute target,
    /// along with the tokens for that attribute target.
    /// See `AttributesData` for more information
    Attributes(AttributesData),
}

impl AttrAnnotatedTokenStream {
    pub fn new(tokens: Vec<(AttrAnnotatedTokenTree, Spacing)>) -> AttrAnnotatedTokenStream {
        AttrAnnotatedTokenStream(Lrc::new(tokens))
    }

    /// Converts this `AttrAnnotatedTokenStream` to a plain `TokenStream
    /// During conversion, `AttrAnnotatedTokenTree::Attributes` get 'flattened'
    /// back to a `TokenStream` of the form `outer_attr attr_target`.
    /// If there are inner attributes, they are inserted into the proper
    /// place in the attribute target tokens.
    pub fn to_tokenstream(&self) -> TokenStream {
        let trees: Vec<_> = self
            .0
            .iter()
            .flat_map(|tree| match &tree.0 {
                AttrAnnotatedTokenTree::Token(inner) => {
                    smallvec![TokenTree::Token(inner.clone(), tree.1)].into_iter()
                }
                AttrAnnotatedTokenTree::Delimited(span, delim, stream) => {
                    smallvec![TokenTree::Delimited(*span, *delim, stream.to_tokenstream()),]
                        .into_iter()
                }
                AttrAnnotatedTokenTree::Attributes(data) => {
                    let mut outer_attrs = Vec::new();
                    let mut inner_attrs = Vec::new();
                    for attr in &data.attrs {
                        match attr.style {
                            crate::AttrStyle::Outer => {
                                outer_attrs.push(attr);
                            }
                            crate::AttrStyle::Inner => {
                                inner_attrs.push(attr);
                            }
                        }
                    }

                    let mut target_tokens: Vec<_> = data
                        .tokens
                        .create_token_stream()
                        .to_tokenstream()
                        .0
                        .iter()
                        .cloned()
                        .collect();
                    if !inner_attrs.is_empty() {
                        let mut found = false;
                        // Check the last two trees (to account for a trailing semi)
                        for tree in target_tokens.iter_mut().rev().take(2) {
                            if let TokenTree::Delimited(span, delim, delim_tokens) = tree {
                                // Inner attributes are only supported on extern blocks, functions, impls,
                                // and modules. All of these have their inner attributes placed at
                                // the beginning of the rightmost outermost braced group:
                                // e.g. fn foo() { #![my_attr} }
                                //
                                // Therefore, we can insert them back into the right location
                                // without needing to do any extra position tracking.
                                //
                                // Note: Outline modules are an exception - they can
                                // have attributes like `#![my_attr]` at the start of a file.
                                // Support for custom attributes in this position is not
                                // properly implemented - we always synthesize fake tokens,
                                // so we never reach this code.

                                let mut builder = TokenStreamBuilder::new();
                                for inner_attr in inner_attrs {
                                    builder.push(inner_attr.tokens().to_tokenstream());
                                }
                                builder.push(delim_tokens.clone());
                                *tree = TokenTree::Delimited(*span, *delim, builder.build());
                                found = true;
                                break;
                            }
                        }

                        assert!(
                            found,
                            "Failed to find trailing delimited group in: {:?}",
                            target_tokens
                        );
                    }
                    let mut flat: SmallVec<[_; 1]> = SmallVec::new();
                    for attr in outer_attrs {
                        // FIXME: Make this more efficient
                        flat.extend(attr.tokens().to_tokenstream().0.clone().iter().cloned());
                    }
                    flat.extend(target_tokens);
                    flat.into_iter()
                }
            })
            .collect();
        TokenStream::new(trees)
    }
}

/// Stores the tokens for an attribute target, along
/// with its attributes.
///
/// This is constructed during parsing when we need to capture
/// tokens.
///
/// For example, `#[cfg(FALSE)] struct Foo {}` would
/// have an `attrs` field containing the `#[cfg(FALSE)]` attr,
/// and a `tokens` field storing the (unparsed) tokens `struct Foo {}`
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct AttributesData {
    /// Attributes, both outer and inner.
    /// These are stored in the original order that they were parsed in.
    pub attrs: AttrVec,
    /// The underlying tokens for the attribute target that `attrs`
    /// are applied to
    pub tokens: LazyTokenStream,
}

/// A `TokenStream` is an abstract sequence of tokens, organized into [`TokenTree`]s.
///
/// The goal is for procedural macros to work with `TokenStream`s and `TokenTree`s
/// instead of a representation of the abstract syntax tree.
/// Today's `TokenTree`s can still contain AST via `token::Interpolated` for
/// backwards compatibility.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct TokenStream(pub(crate) Lrc<Vec<TokenTree>>);

// `TokenStream` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
rustc_data_structures::static_assert_size!(TokenStream, 8);

#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
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
                    (_, TokenTree::Token(Token { kind: token::Comma, .. }, _)) => continue,
                    (
                        TokenTree::Token(token_left, Spacing::Alone),
                        TokenTree::Token(token_right, _),
                    ) if ((token_left.is_ident() && !token_left.is_reserved_ident())
                        || token_left.is_lit())
                        && ((token_right.is_ident() && !token_right.is_reserved_ident())
                            || token_right.is_lit()) =>
                    {
                        token_left.span
                    }
                    (TokenTree::Delimited(sp, ..), _) => sp.entire(),
                    _ => continue,
                };
                let sp = sp.shrink_to_hi();
                let comma = TokenTree::token_alone(token::Comma, sp);
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

impl From<(AttrAnnotatedTokenTree, Spacing)> for AttrAnnotatedTokenStream {
    fn from((tree, spacing): (AttrAnnotatedTokenTree, Spacing)) -> AttrAnnotatedTokenStream {
        AttrAnnotatedTokenStream::new(vec![(tree, spacing)])
    }
}

impl iter::FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(iter: I) -> Self {
        TokenStream::new(iter.into_iter().collect::<Vec<TokenTree>>())
    }
}

impl Eq for TokenStream {}

impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.trees().eq(other.trees())
    }
}

impl TokenStream {
    pub fn new(streams: Vec<TokenTree>) -> TokenStream {
        TokenStream(Lrc::new(streams))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn trees(&self) -> CursorRef<'_> {
        CursorRef::new(self)
    }

    pub fn into_trees(self) -> Cursor {
        Cursor::new(self)
    }

    /// Compares two `TokenStream`s, checking equality without regarding span information.
    pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
        let mut t1 = self.trees();
        let mut t2 = other.trees();
        for (t1, t2) in iter::zip(&mut t1, &mut t2) {
            if !t1.eq_unspanned(&t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    pub fn map_enumerated<F: FnMut(usize, &TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        TokenStream(Lrc::new(self.0.iter().enumerate().map(|(i, tree)| f(i, tree)).collect()))
    }

    fn opt_from_ast(node: &(impl HasAttrs + HasTokens)) -> Option<TokenStream> {
        let tokens = node.tokens()?;
        let attrs = node.attrs();
        let attr_annotated = if attrs.is_empty() {
            tokens.create_token_stream()
        } else {
            let attr_data = AttributesData { attrs: attrs.to_vec().into(), tokens: tokens.clone() };
            AttrAnnotatedTokenStream::new(vec![(
                AttrAnnotatedTokenTree::Attributes(attr_data),
                Spacing::Alone,
            )])
        };
        Some(attr_annotated.to_tokenstream())
    }

    // Create a token stream containing a single token with alone spacing.
    pub fn token_alone(kind: TokenKind, span: Span) -> TokenStream {
        TokenStream::new(vec![TokenTree::token_alone(kind, span)])
    }

    // Create a token stream containing a single token with joint spacing.
    pub fn token_joint(kind: TokenKind, span: Span) -> TokenStream {
        TokenStream::new(vec![TokenTree::token_joint(kind, span)])
    }

    // Create a token stream containing a single `Delimited`.
    pub fn delimited(span: DelimSpan, delim: Delimiter, tts: TokenStream) -> TokenStream {
        TokenStream::new(vec![TokenTree::Delimited(span, delim, tts)])
    }

    pub fn from_ast(node: &(impl HasAttrs + HasSpan + HasTokens + fmt::Debug)) -> TokenStream {
        TokenStream::opt_from_ast(node)
            .unwrap_or_else(|| panic!("missing tokens for node at {:?}: {:?}", node.span(), node))
    }

    pub fn from_nonterminal_ast(nt: &Nonterminal) -> TokenStream {
        match nt {
            Nonterminal::NtIdent(ident, is_raw) => {
                TokenStream::token_alone(token::Ident(ident.name, *is_raw), ident.span)
            }
            Nonterminal::NtLifetime(ident) => {
                TokenStream::token_alone(token::Lifetime(ident.name), ident.span)
            }
            Nonterminal::NtItem(item) => TokenStream::from_ast(item),
            Nonterminal::NtBlock(block) => TokenStream::from_ast(block),
            Nonterminal::NtStmt(stmt) if let StmtKind::Empty = stmt.kind => {
                // FIXME: Properly collect tokens for empty statements.
                TokenStream::token_alone(token::Semi, stmt.span)
            }
            Nonterminal::NtStmt(stmt) => TokenStream::from_ast(stmt),
            Nonterminal::NtPat(pat) => TokenStream::from_ast(pat),
            Nonterminal::NtTy(ty) => TokenStream::from_ast(ty),
            Nonterminal::NtMeta(attr) => TokenStream::from_ast(attr),
            Nonterminal::NtPath(path) => TokenStream::from_ast(path),
            Nonterminal::NtVis(vis) => TokenStream::from_ast(vis),
            Nonterminal::NtExpr(expr) | Nonterminal::NtLiteral(expr) => TokenStream::from_ast(expr),
        }
    }

    fn flatten_token(token: &Token, spacing: Spacing) -> TokenTree {
        match &token.kind {
            token::Interpolated(nt) if let token::NtIdent(ident, is_raw) = **nt => {
                TokenTree::Token(Token::new(token::Ident(ident.name, is_raw), ident.span), spacing)
            }
            token::Interpolated(nt) => TokenTree::Delimited(
                DelimSpan::from_single(token.span),
                Delimiter::Invisible,
                TokenStream::from_nonterminal_ast(&nt).flattened(),
            ),
            _ => TokenTree::Token(token.clone(), spacing),
        }
    }

    fn flatten_token_tree(tree: &TokenTree) -> TokenTree {
        match tree {
            TokenTree::Token(token, spacing) => TokenStream::flatten_token(token, *spacing),
            TokenTree::Delimited(span, delim, tts) => {
                TokenTree::Delimited(*span, *delim, tts.flattened())
            }
        }
    }

    #[must_use]
    pub fn flattened(&self) -> TokenStream {
        fn can_skip(stream: &TokenStream) -> bool {
            stream.trees().all(|tree| match tree {
                TokenTree::Token(token, _) => !matches!(token.kind, token::Interpolated(_)),
                TokenTree::Delimited(_, _, inner) => can_skip(inner),
            })
        }

        if can_skip(self) {
            return self.clone();
        }

        self.trees().map(|tree| TokenStream::flatten_token_tree(tree)).collect()
    }
}

// 99.5%+ of the time we have 1 or 2 elements in this vector.
#[derive(Clone)]
pub struct TokenStreamBuilder(SmallVec<[TokenStream; 2]>);

impl TokenStreamBuilder {
    pub fn new() -> TokenStreamBuilder {
        TokenStreamBuilder(SmallVec::new())
    }

    pub fn push(&mut self, stream: TokenStream) {
        self.0.push(stream);
    }

    pub fn build(self) -> TokenStream {
        let mut streams = self.0;
        match streams.len() {
            0 => TokenStream::default(),
            1 => streams.pop().unwrap(),
            _ => {
                // We will extend the first stream in `streams` with the
                // elements from the subsequent streams. This requires using
                // `make_mut()` on the first stream, and in practice this
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

                // Get the first stream, which will become the result stream.
                // If it's `None`, create an empty stream.
                let mut iter = streams.into_iter();
                let mut res_stream_lrc = iter.next().unwrap().0;

                // Append the subsequent elements to the result stream, after
                // reserving space for them.
                let res_vec_mut = Lrc::make_mut(&mut res_stream_lrc);
                res_vec_mut.reserve(num_appends);
                for stream in iter {
                    let stream_iter = stream.0.iter().cloned();

                    // If (a) `res_mut_vec` is not empty and the last tree
                    // within it is a token tree marked with `Joint`, and (b)
                    // `stream` is not empty and the first tree within it is a
                    // token tree, and (c) the two tokens can be glued
                    // together...
                    if let Some(TokenTree::Token(last_tok, Spacing::Joint)) = res_vec_mut.last()
                        && let Some(TokenTree::Token(tok, spacing)) = stream.0.first()
                        && let Some(glued_tok) = last_tok.glue(&tok)
                    {
                        // ...then overwrite the last token tree in
                        // `res_vec_mut` with the glued token, and skip the
                        // first token tree from `stream`.
                        *res_vec_mut.last_mut().unwrap() = TokenTree::Token(glued_tok, *spacing);
                        res_vec_mut.extend(stream_iter.skip(1));
                    } else {
                        // Append all of `stream`.
                        res_vec_mut.extend(stream_iter);
                    }
                }

                TokenStream(res_stream_lrc)
            }
        }
    }
}

/// By-reference iterator over a [`TokenStream`].
#[derive(Clone)]
pub struct CursorRef<'t> {
    stream: &'t TokenStream,
    index: usize,
}

impl<'t> CursorRef<'t> {
    fn new(stream: &'t TokenStream) -> Self {
        CursorRef { stream, index: 0 }
    }

    pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
        self.stream.0.get(self.index + n)
    }
}

impl<'t> Iterator for CursorRef<'t> {
    type Item = &'t TokenTree;

    fn next(&mut self) -> Option<&'t TokenTree> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree
        })
    }
}

/// Owning by-value iterator over a [`TokenStream`].
// FIXME: Many uses of this can be replaced with by-reference iterator to avoid clones.
#[derive(Clone)]
pub struct Cursor {
    pub stream: TokenStream,
    index: usize,
}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree.clone()
        })
    }
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Cursor { stream, index: 0 }
    }

    #[inline]
    pub fn next_ref(&mut self) -> Option<&TokenTree> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree
        })
    }

    pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
        self.stream.0.get(self.index + n)
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
