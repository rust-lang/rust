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

use std::{fmt, iter, mem};

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
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `Delimited`.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    Delimited(DelimSpan, Delimiter, TokenStream),
}

// Ensure all fields of `TokenTree` are `DynSend` and `DynSync`.
#[cfg(parallel_compiler)]
fn _dummy()
where
    Token: sync::DynSend + sync::DynSync,
    Spacing: sync::DynSend + sync::DynSync,
    DelimSpan: sync::DynSend + sync::DynSync,
    Delimiter: sync::DynSend + sync::DynSync,
    TokenStream: sync::DynSend + sync::DynSync,
{
}

impl TokenTree {
    /// Checks if this `TokenTree` is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (TokenTree::Token(token, _), TokenTree::Token(token2, _)) => token.kind == token2.kind,
            (TokenTree::Delimited(_, delim, tts), TokenTree::Delimited(_, delim2, tts2)) => {
                delim == delim2 && tts.eq_unspanned(tts2)
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

    /// Create a `TokenTree::Token` with alone spacing.
    pub fn token_alone(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span), Spacing::Alone)
    }

    /// Create a `TokenTree::Token` with joint spacing.
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

pub trait ToAttrTokenStream: sync::DynSend + sync::DynSync {
    fn to_attr_token_stream(&self) -> AttrTokenStream;
}

impl ToAttrTokenStream for AttrTokenStream {
    fn to_attr_token_stream(&self) -> AttrTokenStream {
        self.clone()
    }
}

/// A lazy version of [`TokenStream`], which defers creation
/// of an actual `TokenStream` until it is needed.
/// `Box` is here only to reduce the structure size.
#[derive(Clone)]
pub struct LazyAttrTokenStream(Lrc<Box<dyn ToAttrTokenStream>>);

impl LazyAttrTokenStream {
    pub fn new(inner: impl ToAttrTokenStream + 'static) -> LazyAttrTokenStream {
        LazyAttrTokenStream(Lrc::new(Box::new(inner)))
    }

    pub fn to_attr_token_stream(&self) -> AttrTokenStream {
        self.0.to_attr_token_stream()
    }
}

impl fmt::Debug for LazyAttrTokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyAttrTokenStream({:?})", self.to_attr_token_stream())
    }
}

impl<S: Encoder> Encodable<S> for LazyAttrTokenStream {
    fn encode(&self, s: &mut S) {
        // Used by AST json printing.
        Encodable::encode(&self.to_attr_token_stream(), s);
    }
}

impl<D: Decoder> Decodable<D> for LazyAttrTokenStream {
    fn decode(_d: &mut D) -> Self {
        panic!("Attempted to decode LazyAttrTokenStream");
    }
}

impl<CTX> HashStable<CTX> for LazyAttrTokenStream {
    fn hash_stable(&self, _hcx: &mut CTX, _hasher: &mut StableHasher) {
        panic!("Attempted to compute stable hash for LazyAttrTokenStream");
    }
}

/// An `AttrTokenStream` is similar to a `TokenStream`, but with extra
/// information about the tokens for attribute targets. This is used
/// during expansion to perform early cfg-expansion, and to process attributes
/// during proc-macro invocations.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct AttrTokenStream(pub Lrc<Vec<AttrTokenTree>>);

/// Like `TokenTree`, but for `AttrTokenStream`.
#[derive(Clone, Debug, Encodable, Decodable)]
pub enum AttrTokenTree {
    Token(Token, Spacing),
    Delimited(DelimSpan, Delimiter, AttrTokenStream),
    /// Stores the attributes for an attribute target,
    /// along with the tokens for that attribute target.
    /// See `AttributesData` for more information
    Attributes(AttributesData),
}

impl AttrTokenStream {
    pub fn new(tokens: Vec<AttrTokenTree>) -> AttrTokenStream {
        AttrTokenStream(Lrc::new(tokens))
    }

    /// Converts this `AttrTokenStream` to a plain `TokenStream`.
    /// During conversion, `AttrTokenTree::Attributes` get 'flattened'
    /// back to a `TokenStream` of the form `outer_attr attr_target`.
    /// If there are inner attributes, they are inserted into the proper
    /// place in the attribute target tokens.
    pub fn to_tokenstream(&self) -> TokenStream {
        let trees: Vec<_> = self
            .0
            .iter()
            .flat_map(|tree| match &tree {
                AttrTokenTree::Token(inner, spacing) => {
                    smallvec![TokenTree::Token(inner.clone(), *spacing)].into_iter()
                }
                AttrTokenTree::Delimited(span, delim, stream) => {
                    smallvec![TokenTree::Delimited(*span, *delim, stream.to_tokenstream()),]
                        .into_iter()
                }
                AttrTokenTree::Attributes(data) => {
                    let mut outer_attrs = Vec::new();
                    let mut inner_attrs = Vec::new();
                    for attr in &data.attrs {
                        match attr.style {
                            crate::AttrStyle::Outer => outer_attrs.push(attr),
                            crate::AttrStyle::Inner => inner_attrs.push(attr),
                        }
                    }

                    let mut target_tokens: Vec<_> = data
                        .tokens
                        .to_attr_token_stream()
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
                                // Inner attributes are only supported on extern blocks, functions,
                                // impls, and modules. All of these have their inner attributes
                                // placed at the beginning of the rightmost outermost braced group:
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

                                let mut stream = TokenStream::default();
                                for inner_attr in inner_attrs {
                                    stream.push_stream(inner_attr.tokens());
                                }
                                stream.push_stream(delim_tokens.clone());
                                *tree = TokenTree::Delimited(*span, *delim, stream);
                                found = true;
                                break;
                            }
                        }

                        assert!(
                            found,
                            "Failed to find trailing delimited group in: {target_tokens:?}"
                        );
                    }
                    let mut flat: SmallVec<[_; 1]> = SmallVec::new();
                    for attr in outer_attrs {
                        // FIXME: Make this more efficient
                        flat.extend(attr.tokens().0.clone().iter().cloned());
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
    pub tokens: LazyAttrTokenStream,
}

/// A `TokenStream` is an abstract sequence of tokens, organized into [`TokenTree`]s.
///
/// The goal is for procedural macros to work with `TokenStream`s and `TokenTree`s
/// instead of a representation of the abstract syntax tree.
/// Today's `TokenTree`s can still contain AST via `token::Interpolated` for
/// backwards compatibility.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct TokenStream(pub(crate) Lrc<Vec<TokenTree>>);

/// Similar to `proc_macro::Spacing`, but for tokens.
///
/// Note that all `ast::TokenTree::Token` instances have a `Spacing`, but when
/// we convert to `proc_macro::TokenTree` for proc macros only `Punct`
/// `TokenTree`s have a `proc_macro::Spacing`.
#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum Spacing {
    /// The token is not immediately followed by an operator token (as
    /// determined by `Token::is_op`). E.g. a `+` token is `Alone` in `+ =`,
    /// `+/*foo*/=`, `+ident`, and `+()`.
    Alone,

    /// The token is immediately followed by an operator token. E.g. a `+`
    /// token is `Joint` in `+=` and `++`.
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

impl FromIterator<TokenTree> for TokenStream {
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

    pub fn trees(&self) -> RefTokenTreeCursor<'_> {
        RefTokenTreeCursor::new(self)
    }

    pub fn into_trees(self) -> TokenTreeCursor {
        TokenTreeCursor::new(self)
    }

    /// Compares two `TokenStream`s, checking equality without regarding span information.
    pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
        let mut t1 = self.trees();
        let mut t2 = other.trees();
        for (t1, t2) in iter::zip(&mut t1, &mut t2) {
            if !t1.eq_unspanned(t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    /// Applies the supplied function to each `TokenTree` and its index in `self`, returning a new `TokenStream`
    ///
    /// It is equivalent to `TokenStream::new(self.trees().cloned().enumerate().map(|(i, tt)| f(i, tt)).collect())`.
    pub fn map_enumerated_owned(
        mut self,
        mut f: impl FnMut(usize, TokenTree) -> TokenTree,
    ) -> TokenStream {
        let owned = Lrc::make_mut(&mut self.0); // clone if necessary
        // rely on vec's in-place optimizations to avoid another allocation
        *owned = mem::take(owned).into_iter().enumerate().map(|(i, tree)| f(i, tree)).collect();
        self
    }

    /// Create a token stream containing a single token with alone spacing.
    pub fn token_alone(kind: TokenKind, span: Span) -> TokenStream {
        TokenStream::new(vec![TokenTree::token_alone(kind, span)])
    }

    /// Create a token stream containing a single token with joint spacing.
    pub fn token_joint(kind: TokenKind, span: Span) -> TokenStream {
        TokenStream::new(vec![TokenTree::token_joint(kind, span)])
    }

    /// Create a token stream containing a single `Delimited`.
    pub fn delimited(span: DelimSpan, delim: Delimiter, tts: TokenStream) -> TokenStream {
        TokenStream::new(vec![TokenTree::Delimited(span, delim, tts)])
    }

    pub fn from_ast(node: &(impl HasAttrs + HasSpan + HasTokens + fmt::Debug)) -> TokenStream {
        let Some(tokens) = node.tokens() else {
            panic!("missing tokens for node at {:?}: {:?}", node.span(), node);
        };
        let attrs = node.attrs();
        let attr_stream = if attrs.is_empty() {
            tokens.to_attr_token_stream()
        } else {
            let attr_data =
                AttributesData { attrs: attrs.iter().cloned().collect(), tokens: tokens.clone() };
            AttrTokenStream::new(vec![AttrTokenTree::Attributes(attr_data)])
        };
        attr_stream.to_tokenstream()
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
                TokenStream::from_nonterminal_ast(nt).flattened(),
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

    // If `vec` is not empty, try to glue `tt` onto its last token. The return
    // value indicates if gluing took place.
    fn try_glue_to_last(vec: &mut Vec<TokenTree>, tt: &TokenTree) -> bool {
        if let Some(TokenTree::Token(last_tok, Spacing::Joint)) = vec.last()
            && let TokenTree::Token(tok, spacing) = tt
            && let Some(glued_tok) = last_tok.glue(tok)
        {
            // ...then overwrite the last token tree in `vec` with the
            // glued token, and skip the first token tree from `stream`.
            *vec.last_mut().unwrap() = TokenTree::Token(glued_tok, *spacing);
            true
        } else {
            false
        }
    }

    /// Push `tt` onto the end of the stream, possibly gluing it to the last
    /// token. Uses `make_mut` to maximize efficiency.
    pub fn push_tree(&mut self, tt: TokenTree) {
        let vec_mut = Lrc::make_mut(&mut self.0);

        if Self::try_glue_to_last(vec_mut, &tt) {
            // nothing else to do
        } else {
            vec_mut.push(tt);
        }
    }

    /// Push `stream` onto the end of the stream, possibly gluing the first
    /// token tree to the last token. (No other token trees will be glued.)
    /// Uses `make_mut` to maximize efficiency.
    pub fn push_stream(&mut self, stream: TokenStream) {
        let vec_mut = Lrc::make_mut(&mut self.0);

        let stream_iter = stream.0.iter().cloned();

        if let Some(first) = stream.0.first() && Self::try_glue_to_last(vec_mut, first) {
            // Now skip the first token tree from `stream`.
            vec_mut.extend(stream_iter.skip(1));
        } else {
            // Append all of `stream`.
            vec_mut.extend(stream_iter);
        }
    }

    pub fn chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, TokenTree> {
        self.0.chunks(chunk_size)
    }
}

/// By-reference iterator over a [`TokenStream`], that produces `&TokenTree`
/// items.
#[derive(Clone)]
pub struct RefTokenTreeCursor<'t> {
    stream: &'t TokenStream,
    index: usize,
}

impl<'t> RefTokenTreeCursor<'t> {
    fn new(stream: &'t TokenStream) -> Self {
        RefTokenTreeCursor { stream, index: 0 }
    }

    pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
        self.stream.0.get(self.index + n)
    }
}

impl<'t> Iterator for RefTokenTreeCursor<'t> {
    type Item = &'t TokenTree;

    fn next(&mut self) -> Option<&'t TokenTree> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree
        })
    }
}

/// Owning by-value iterator over a [`TokenStream`], that produces `TokenTree`
/// items.
// FIXME: Many uses of this can be replaced with by-reference iterator to avoid clones.
#[derive(Clone)]
pub struct TokenTreeCursor {
    pub stream: TokenStream,
    index: usize,
}

impl Iterator for TokenTreeCursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree.clone()
        })
    }
}

impl TokenTreeCursor {
    fn new(stream: TokenStream) -> Self {
        TokenTreeCursor { stream, index: 0 }
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

    // Replace the previously obtained token tree with `tts`, and rewind to
    // just before them.
    pub fn replace_prev_and_rewind(&mut self, tts: Vec<TokenTree>) {
        assert!(self.index > 0);
        self.index -= 1;
        let stream = Lrc::make_mut(&mut self.stream.0);
        stream.splice(self.index..self.index + 1, tts);
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

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(AttrTokenStream, 8);
    static_assert_size!(AttrTokenTree, 32);
    static_assert_size!(LazyAttrTokenStream, 8);
    static_assert_size!(TokenStream, 8);
    static_assert_size!(TokenTree, 32);
    // tidy-alphabetical-end
}
