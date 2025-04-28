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

use std::borrow::Cow;
use std::sync::Arc;
use std::{cmp, fmt, iter};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{DUMMY_SP, Span, SpanDecoder, SpanEncoder, Symbol, sym};

use crate::ast::AttrStyle;
use crate::ast_traits::{HasAttrs, HasTokens};
use crate::token::{self, Delimiter, Token, TokenKind};
use crate::{AttrVec, Attribute};

/// Part of a `TokenStream`.
#[derive(Debug, Clone, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum TokenTree {
    /// A single token. Should never be `OpenDelim` or `CloseDelim`, because
    /// delimiters are implicitly represented by `Delimited`.
    Token(Token, Spacing),
    /// A delimited sequence of token trees.
    Delimited(DelimSpan, DelimSpacing, Delimiter, TokenStream),
}

// Ensure all fields of `TokenTree` are `DynSend` and `DynSync`.
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
    /// Checks if this `TokenTree` is equal to the other, regardless of span/spacing information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (TokenTree::Token(token, _), TokenTree::Token(token2, _)) => token.kind == token2.kind,
            (TokenTree::Delimited(.., delim, tts), TokenTree::Delimited(.., delim2, tts2)) => {
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

    /// Create a `TokenTree::Token` with alone spacing.
    pub fn token_alone(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span), Spacing::Alone)
    }

    /// Create a `TokenTree::Token` with joint spacing.
    pub fn token_joint(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span), Spacing::Joint)
    }

    /// Create a `TokenTree::Token` with joint-hidden spacing.
    pub fn token_joint_hidden(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span), Spacing::JointHidden)
    }

    pub fn uninterpolate(&self) -> Cow<'_, TokenTree> {
        match self {
            TokenTree::Token(token, spacing) => match token.uninterpolate() {
                Cow::Owned(token) => Cow::Owned(TokenTree::Token(token, *spacing)),
                Cow::Borrowed(_) => Cow::Borrowed(self),
            },
            _ => Cow::Borrowed(self),
        }
    }
}

impl<CTX> HashStable<CTX> for TokenStream
where
    CTX: crate::HashStableContext,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        for sub_tt in self.iter() {
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
pub struct LazyAttrTokenStream(Arc<Box<dyn ToAttrTokenStream>>);

impl LazyAttrTokenStream {
    pub fn new(inner: impl ToAttrTokenStream + 'static) -> LazyAttrTokenStream {
        LazyAttrTokenStream(Arc::new(Box::new(inner)))
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

impl<S: SpanEncoder> Encodable<S> for LazyAttrTokenStream {
    fn encode(&self, _s: &mut S) {
        panic!("Attempted to encode LazyAttrTokenStream");
    }
}

impl<D: SpanDecoder> Decodable<D> for LazyAttrTokenStream {
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
pub struct AttrTokenStream(pub Arc<Vec<AttrTokenTree>>);

/// Like `TokenTree`, but for `AttrTokenStream`.
#[derive(Clone, Debug, Encodable, Decodable)]
pub enum AttrTokenTree {
    Token(Token, Spacing),
    Delimited(DelimSpan, DelimSpacing, Delimiter, AttrTokenStream),
    /// Stores the attributes for an attribute target,
    /// along with the tokens for that attribute target.
    /// See `AttrsTarget` for more information
    AttrsTarget(AttrsTarget),
}

impl AttrTokenStream {
    pub fn new(tokens: Vec<AttrTokenTree>) -> AttrTokenStream {
        AttrTokenStream(Arc::new(tokens))
    }

    /// Converts this `AttrTokenStream` to a plain `Vec<TokenTree>`. During
    /// conversion, any `AttrTokenTree::AttrsTarget` gets "flattened" back to a
    /// `TokenStream`, as described in the comment on
    /// `attrs_and_tokens_to_token_trees`.
    pub fn to_token_trees(&self) -> Vec<TokenTree> {
        let mut res = Vec::with_capacity(self.0.len());
        for tree in self.0.iter() {
            match tree {
                AttrTokenTree::Token(inner, spacing) => {
                    res.push(TokenTree::Token(inner.clone(), *spacing));
                }
                AttrTokenTree::Delimited(span, spacing, delim, stream) => {
                    res.push(TokenTree::Delimited(
                        *span,
                        *spacing,
                        *delim,
                        TokenStream::new(stream.to_token_trees()),
                    ))
                }
                AttrTokenTree::AttrsTarget(target) => {
                    attrs_and_tokens_to_token_trees(&target.attrs, &target.tokens, &mut res);
                }
            }
        }
        res
    }
}

// Converts multiple attributes and the tokens for a target AST node into token trees, and appends
// them to `res`.
//
// Example: if the AST node is "fn f() { blah(); }", then:
// - Simple if no attributes are present, e.g. "fn f() { blah(); }"
// - Simple if only outer attribute are present, e.g. "#[outer1] #[outer2] fn f() { blah(); }"
// - Trickier if inner attributes are present, because they must be moved within the AST node's
//   tokens, e.g. "#[outer] fn f() { #![inner] blah() }"
fn attrs_and_tokens_to_token_trees(
    attrs: &[Attribute],
    target_tokens: &LazyAttrTokenStream,
    res: &mut Vec<TokenTree>,
) {
    let idx = attrs.partition_point(|attr| matches!(attr.style, crate::AttrStyle::Outer));
    let (outer_attrs, inner_attrs) = attrs.split_at(idx);

    // Add outer attribute tokens.
    for attr in outer_attrs {
        res.extend(attr.token_trees());
    }

    // Add target AST node tokens.
    res.extend(target_tokens.to_attr_token_stream().to_token_trees());

    // Insert inner attribute tokens.
    if !inner_attrs.is_empty() {
        let found = insert_inner_attrs(inner_attrs, res);
        assert!(found, "Failed to find trailing delimited group in: {res:?}");
    }

    // Inner attributes are only supported on blocks, functions, impls, and
    // modules. All of these have their inner attributes placed at the
    // beginning of the rightmost outermost braced group:
    // e.g. `fn foo() { #![my_attr] }`. (Note: the braces may be within
    // invisible delimiters.)
    //
    // Therefore, we can insert them back into the right location without
    // needing to do any extra position tracking.
    //
    // Note: Outline modules are an exception - they can have attributes like
    // `#![my_attr]` at the start of a file. Support for custom attributes in
    // this position is not properly implemented - we always synthesize fake
    // tokens, so we never reach this code.
    fn insert_inner_attrs(inner_attrs: &[Attribute], tts: &mut Vec<TokenTree>) -> bool {
        for tree in tts.iter_mut().rev() {
            if let TokenTree::Delimited(span, spacing, Delimiter::Brace, stream) = tree {
                // Found it: the rightmost, outermost braced group.
                let mut tts = vec![];
                for inner_attr in inner_attrs {
                    tts.extend(inner_attr.token_trees());
                }
                tts.extend(stream.0.iter().cloned());
                let stream = TokenStream::new(tts);
                *tree = TokenTree::Delimited(*span, *spacing, Delimiter::Brace, stream);
                return true;
            } else if let TokenTree::Delimited(span, spacing, Delimiter::Invisible(src), stream) =
                tree
            {
                // Recurse inside invisible delimiters.
                let mut vec: Vec<_> = stream.iter().cloned().collect();
                if insert_inner_attrs(inner_attrs, &mut vec) {
                    *tree = TokenTree::Delimited(
                        *span,
                        *spacing,
                        Delimiter::Invisible(*src),
                        TokenStream::new(vec),
                    );
                    return true;
                }
            }
        }
        false
    }
}

/// Stores the tokens for an attribute target, along
/// with its attributes.
///
/// This is constructed during parsing when we need to capture
/// tokens, for `cfg` and `cfg_attr` attributes.
///
/// For example, `#[cfg(FALSE)] struct Foo {}` would
/// have an `attrs` field containing the `#[cfg(FALSE)]` attr,
/// and a `tokens` field storing the (unparsed) tokens `struct Foo {}`
///
/// The `cfg`/`cfg_attr` processing occurs in
/// `StripUnconfigured::configure_tokens`.
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct AttrsTarget {
    /// Attributes, both outer and inner.
    /// These are stored in the original order that they were parsed in.
    pub attrs: AttrVec,
    /// The underlying tokens for the attribute target that `attrs`
    /// are applied to
    pub tokens: LazyAttrTokenStream,
}

/// A `TokenStream` is an abstract sequence of tokens, organized into [`TokenTree`]s.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct TokenStream(pub(crate) Arc<Vec<TokenTree>>);

/// Indicates whether a token can join with the following token to form a
/// compound token. Used for conversions to `proc_macro::Spacing`. Also used to
/// guide pretty-printing, which is where the `JointHidden` value (which isn't
/// part of `proc_macro::Spacing`) comes in useful.
#[derive(Clone, Copy, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub enum Spacing {
    /// The token cannot join with the following token to form a compound
    /// token.
    ///
    /// In token streams parsed from source code, the compiler will use `Alone`
    /// for any token immediately followed by whitespace, a non-doc comment, or
    /// EOF.
    ///
    /// When constructing token streams within the compiler, use this for each
    /// token that (a) should be pretty-printed with a space after it, or (b)
    /// is the last token in the stream. (In the latter case the choice of
    /// spacing doesn't matter because it is never used for the last token. We
    /// arbitrarily use `Alone`.)
    ///
    /// Converts to `proc_macro::Spacing::Alone`, and
    /// `proc_macro::Spacing::Alone` converts back to this.
    Alone,

    /// The token can join with the following token to form a compound token.
    ///
    /// In token streams parsed from source code, the compiler will use `Joint`
    /// for any token immediately followed by punctuation (as determined by
    /// `Token::is_punct`).
    ///
    /// When constructing token streams within the compiler, use this for each
    /// token that (a) should be pretty-printed without a space after it, and
    /// (b) is followed by a punctuation token.
    ///
    /// Converts to `proc_macro::Spacing::Joint`, and
    /// `proc_macro::Spacing::Joint` converts back to this.
    Joint,

    /// The token can join with the following token to form a compound token,
    /// but this will not be visible at the proc macro level. (This is what the
    /// `Hidden` means; see below.)
    ///
    /// In token streams parsed from source code, the compiler will use
    /// `JointHidden` for any token immediately followed by anything not
    /// covered by the `Alone` and `Joint` cases: an identifier, lifetime,
    /// literal, delimiter, doc comment.
    ///
    /// When constructing token streams, use this for each token that (a)
    /// should be pretty-printed without a space after it, and (b) is followed
    /// by a non-punctuation token.
    ///
    /// Converts to `proc_macro::Spacing::Alone`, but
    /// `proc_macro::Spacing::Alone` converts back to `token::Spacing::Alone`.
    /// Because of that, pretty-printing of `TokenStream`s produced by proc
    /// macros is unavoidably uglier (with more whitespace between tokens) than
    /// pretty-printing of `TokenStream`'s produced by other means (i.e. parsed
    /// source code, internally constructed token streams, and token streams
    /// produced by declarative macros).
    JointHidden,
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
        self.iter().eq(other.iter())
    }
}

impl TokenStream {
    pub fn new(tts: Vec<TokenTree>) -> TokenStream {
        TokenStream(Arc::new(tts))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, index: usize) -> Option<&TokenTree> {
        self.0.get(index)
    }

    pub fn iter(&self) -> TokenStreamIter<'_> {
        TokenStreamIter::new(self)
    }

    /// Compares two `TokenStream`s, checking equality without regarding span information.
    pub fn eq_unspanned(&self, other: &TokenStream) -> bool {
        let mut iter1 = self.iter();
        let mut iter2 = other.iter();
        for (tt1, tt2) in iter::zip(&mut iter1, &mut iter2) {
            if !tt1.eq_unspanned(tt2) {
                return false;
            }
        }
        iter1.next().is_none() && iter2.next().is_none()
    }

    /// Create a token stream containing a single token with alone spacing. The
    /// spacing used for the final token in a constructed stream doesn't matter
    /// because it's never used. In practice we arbitrarily use
    /// `Spacing::Alone`.
    pub fn token_alone(kind: TokenKind, span: Span) -> TokenStream {
        TokenStream::new(vec![TokenTree::token_alone(kind, span)])
    }

    pub fn from_ast(node: &(impl HasAttrs + HasTokens + fmt::Debug)) -> TokenStream {
        let tokens = node.tokens().unwrap_or_else(|| panic!("missing tokens for node: {:?}", node));
        let mut tts = vec![];
        attrs_and_tokens_to_token_trees(node.attrs(), tokens, &mut tts);
        TokenStream::new(tts)
    }

    // If `vec` is not empty, try to glue `tt` onto its last token. The return
    // value indicates if gluing took place.
    fn try_glue_to_last(vec: &mut Vec<TokenTree>, tt: &TokenTree) -> bool {
        if let Some(TokenTree::Token(last_tok, Spacing::Joint | Spacing::JointHidden)) = vec.last()
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
        let vec_mut = Arc::make_mut(&mut self.0);

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
        let vec_mut = Arc::make_mut(&mut self.0);

        let stream_iter = stream.0.iter().cloned();

        if let Some(first) = stream.0.first()
            && Self::try_glue_to_last(vec_mut, first)
        {
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

    /// Desugar doc comments like `/// foo` in the stream into `#[doc =
    /// r"foo"]`. Modifies the `TokenStream` via `Arc::make_mut`, but as little
    /// as possible.
    pub fn desugar_doc_comments(&mut self) {
        if let Some(desugared_stream) = desugar_inner(self.clone()) {
            *self = desugared_stream;
        }

        // The return value is `None` if nothing in `stream` changed.
        fn desugar_inner(mut stream: TokenStream) -> Option<TokenStream> {
            let mut i = 0;
            let mut modified = false;
            while let Some(tt) = stream.0.get(i) {
                match tt {
                    &TokenTree::Token(
                        Token { kind: token::DocComment(_, attr_style, data), span },
                        _spacing,
                    ) => {
                        let desugared = desugared_tts(attr_style, data, span);
                        let desugared_len = desugared.len();
                        Arc::make_mut(&mut stream.0).splice(i..i + 1, desugared);
                        modified = true;
                        i += desugared_len;
                    }

                    &TokenTree::Token(..) => i += 1,

                    &TokenTree::Delimited(sp, spacing, delim, ref delim_stream) => {
                        if let Some(desugared_delim_stream) = desugar_inner(delim_stream.clone()) {
                            let new_tt =
                                TokenTree::Delimited(sp, spacing, delim, desugared_delim_stream);
                            Arc::make_mut(&mut stream.0)[i] = new_tt;
                            modified = true;
                        }
                        i += 1;
                    }
                }
            }
            if modified { Some(stream) } else { None }
        }

        fn desugared_tts(attr_style: AttrStyle, data: Symbol, span: Span) -> Vec<TokenTree> {
            // Searches for the occurrences of `"#*` and returns the minimum number of `#`s
            // required to wrap the text. E.g.
            // - `abc d` is wrapped as `r"abc d"` (num_of_hashes = 0)
            // - `abc "d"` is wrapped as `r#"abc "d""#` (num_of_hashes = 1)
            // - `abc "##d##"` is wrapped as `r###"abc ##"d"##"###` (num_of_hashes = 3)
            let mut num_of_hashes = 0;
            let mut count = 0;
            for ch in data.as_str().chars() {
                count = match ch {
                    '"' => 1,
                    '#' if count > 0 => count + 1,
                    _ => 0,
                };
                num_of_hashes = cmp::max(num_of_hashes, count);
            }

            // `/// foo` becomes `[doc = r"foo"]`.
            let delim_span = DelimSpan::from_single(span);
            let body = TokenTree::Delimited(
                delim_span,
                DelimSpacing::new(Spacing::JointHidden, Spacing::Alone),
                Delimiter::Bracket,
                [
                    TokenTree::token_alone(token::Ident(sym::doc, token::IdentIsRaw::No), span),
                    TokenTree::token_alone(token::Eq, span),
                    TokenTree::token_alone(
                        TokenKind::lit(token::StrRaw(num_of_hashes), data, None),
                        span,
                    ),
                ]
                .into_iter()
                .collect::<TokenStream>(),
            );

            if attr_style == AttrStyle::Inner {
                vec![
                    TokenTree::token_joint(token::Pound, span),
                    TokenTree::token_joint_hidden(token::Bang, span),
                    body,
                ]
            } else {
                vec![TokenTree::token_joint_hidden(token::Pound, span), body]
            }
        }
    }
}

#[derive(Clone)]
pub struct TokenStreamIter<'t> {
    stream: &'t TokenStream,
    index: usize,
}

impl<'t> TokenStreamIter<'t> {
    fn new(stream: &'t TokenStream) -> Self {
        TokenStreamIter { stream, index: 0 }
    }

    // Peeking could be done via `Peekable`, but most iterators need peeking,
    // and this is simple and avoids the need to use `peekable` and `Peekable`
    // at all the use sites.
    pub fn peek(&self) -> Option<&'t TokenTree> {
        self.stream.0.get(self.index)
    }
}

impl<'t> Iterator for TokenStreamIter<'t> {
    type Item = &'t TokenTree;

    fn next(&mut self) -> Option<&'t TokenTree> {
        self.stream.0.get(self.index).map(|tree| {
            self.index += 1;
            tree
        })
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

#[derive(Copy, Clone, Debug, PartialEq, Encodable, Decodable, HashStable_Generic)]
pub struct DelimSpacing {
    pub open: Spacing,
    pub close: Spacing,
}

impl DelimSpacing {
    pub fn new(open: Spacing, close: Spacing) -> DelimSpacing {
        DelimSpacing { open, close }
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(AttrTokenStream, 8);
    static_assert_size!(AttrTokenTree, 32);
    static_assert_size!(LazyAttrTokenStream, 8);
    static_assert_size!(Option<LazyAttrTokenStream>, 8); // must be small, used in many AST nodes
    static_assert_size!(TokenStream, 8);
    static_assert_size!(TokenTree, 32);
    // tidy-alphabetical-end
}
