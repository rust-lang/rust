//! # Token Streams
//!
//! `TokenStream`s represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence of [`TokenTree`]s,
//! which are themselves a single [`Token`] or a `Delimited` subsequence of tokens.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;
use std::{cmp, fmt, iter, mem};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::sync;
use rustc_macros::{Decodable, Encodable, HashStable_Generic, Walkable};
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{DUMMY_SP, Span, SpanDecoder, SpanEncoder, Symbol, sym};
use thin_vec::ThinVec;

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
                delim == delim2 && tts.iter().eq_by(tts2.iter(), |a, b| a.eq_unspanned(b))
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

/// A lazy version of [`AttrTokenStream`], which defers creation of an actual
/// `AttrTokenStream` until it is needed.
#[derive(Clone)]
pub struct LazyAttrTokenStream(Arc<LazyAttrTokenStreamInner>);

impl LazyAttrTokenStream {
    pub fn new_direct(stream: AttrTokenStream) -> LazyAttrTokenStream {
        LazyAttrTokenStream(Arc::new(LazyAttrTokenStreamInner::Direct(stream)))
    }

    pub fn new_pending(
        start_token: (Token, Spacing),
        cursor_snapshot: TokenCursor,
        num_calls: u32,
        break_last_token: u32,
        node_replacements: ThinVec<NodeReplacement>,
    ) -> LazyAttrTokenStream {
        LazyAttrTokenStream(Arc::new(LazyAttrTokenStreamInner::Pending {
            start_token,
            cursor_snapshot,
            num_calls,
            break_last_token,
            node_replacements,
        }))
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

/// A token range within a `Parser`'s full token stream.
#[derive(Clone, Debug)]
pub struct ParserRange(pub Range<u32>);

/// A token range within an individual AST node's (lazy) token stream, i.e.
/// relative to that node's first token. Distinct from `ParserRange` so the two
/// kinds of range can't be mixed up.
#[derive(Clone, Debug)]
pub struct NodeRange(pub Range<u32>);

/// Indicates a range of tokens that should be replaced by an `AttrsTarget`
/// (replacement) or be replaced by nothing (deletion). This is used in two
/// places during token collection.
///
/// 1. Replacement. During the parsing of an AST node that may have a
///    `#[derive]` attribute, when we parse a nested AST node that has `#[cfg]`
///    or `#[cfg_attr]`, we replace the entire inner AST node with
///    `FlatToken::AttrsTarget`. This lets us perform eager cfg-expansion on an
///    `AttrTokenStream`.
///
/// 2. Deletion. We delete inner attributes from all collected token streams,
///    and instead track them through the `attrs` field on the AST node. This
///    lets us manipulate them similarly to outer attributes. When we create a
///    `TokenStream`, the inner attributes are inserted into the proper place
///    in the token stream.
///
/// Each replacement starts off in `ParserReplacement` form but is converted to
/// `NodeReplacement` form when it is attached to a single AST node, via
/// `LazyAttrTokenStreamImpl`.
pub type ParserReplacement = (ParserRange, Option<AttrsTarget>);

/// See the comment on `ParserReplacement`.
pub type NodeReplacement = (NodeRange, Option<AttrsTarget>);

impl NodeRange {
    // Converts a range within a parser's tokens to a range within a
    // node's tokens beginning at `start_pos`.
    //
    // For example, imagine a parser with 50 tokens in its token stream, a
    // function that spans `ParserRange(20..40)` and an inner attribute within
    // that function that spans `ParserRange(30..35)`. We would find the inner
    // attribute's range within the function's tokens by subtracting 20, which
    // is the position of the function's start token. This gives
    // `NodeRange(10..15)`.
    pub fn new(ParserRange(parser_range): ParserRange, start_pos: u32) -> NodeRange {
        assert!(!parser_range.is_empty());
        assert!(parser_range.start >= start_pos);
        NodeRange((parser_range.start - start_pos)..(parser_range.end - start_pos))
    }
}

enum LazyAttrTokenStreamInner {
    // The token stream has already been produced.
    Direct(AttrTokenStream),

    // From a value of this type we can reconstruct the `TokenStream` seen by
    // the `f` callback passed to a call to `Parser::collect_tokens`, by
    // replaying the getting of the tokens. This saves us producing a
    // `TokenStream` if it is never needed, e.g. a captured `macro_rules!`
    // argument that is never passed to a proc macro. In practice, token stream
    // creation happens rarely compared to calls to `collect_tokens` (see some
    // statistics in #78736) so we are doing as little up-front work as
    // possible.
    //
    // This also makes `Parser` very cheap to clone, since there is no
    // intermediate collection buffer to clone.
    Pending {
        start_token: (Token, Spacing),
        cursor_snapshot: TokenCursor,
        num_calls: u32,
        break_last_token: u32,
        node_replacements: ThinVec<NodeReplacement>,
    },
}

impl LazyAttrTokenStreamInner {
    fn to_attr_token_stream(&self) -> AttrTokenStream {
        match self {
            LazyAttrTokenStreamInner::Direct(stream) => stream.clone(),
            LazyAttrTokenStreamInner::Pending {
                start_token,
                cursor_snapshot,
                num_calls,
                break_last_token,
                node_replacements,
            } => {
                // The token produced by the final call to `{,inlined_}next` was not
                // actually consumed by the callback. The combination of chaining the
                // initial token and using `take` produces the desired result - we
                // produce an empty `TokenStream` if no calls were made, and omit the
                // final token otherwise.
                let mut cursor_snapshot = cursor_snapshot.clone();
                let tokens = iter::once(FlatToken::Token(*start_token))
                    .chain(iter::repeat_with(|| FlatToken::Token(cursor_snapshot.next())))
                    .take(*num_calls as usize);

                if node_replacements.is_empty() {
                    make_attr_token_stream(tokens, *break_last_token)
                } else {
                    let mut tokens: Vec<_> = tokens.collect();
                    let mut node_replacements = node_replacements.to_vec();
                    node_replacements.sort_by_key(|(range, _)| range.0.start);

                    #[cfg(debug_assertions)]
                    for [(node_range, tokens), (next_node_range, next_tokens)] in
                        node_replacements.array_windows()
                    {
                        assert!(
                            node_range.0.end <= next_node_range.0.start
                                || node_range.0.end >= next_node_range.0.end,
                            "Node ranges should be disjoint or nested: ({:?}, {:?}) ({:?}, {:?})",
                            node_range,
                            tokens,
                            next_node_range,
                            next_tokens,
                        );
                    }

                    // Process the replace ranges, starting from the highest start
                    // position and working our way back. If have tokens like:
                    //
                    // `#[cfg(FALSE)] struct Foo { #[cfg(FALSE)] field: bool }`
                    //
                    // Then we will generate replace ranges for both
                    // the `#[cfg(FALSE)] field: bool` and the entire
                    // `#[cfg(FALSE)] struct Foo { #[cfg(FALSE)] field: bool }`
                    //
                    // By starting processing from the replace range with the greatest
                    // start position, we ensure that any (outer) replace range which
                    // encloses another (inner) replace range will fully overwrite the
                    // inner range's replacement.
                    for (node_range, target) in node_replacements.into_iter().rev() {
                        assert!(
                            !node_range.0.is_empty(),
                            "Cannot replace an empty node range: {:?}",
                            node_range.0
                        );

                        // Replace the tokens in range with zero or one `FlatToken::AttrsTarget`s,
                        // plus enough `FlatToken::Empty`s to fill up the rest of the range. This
                        // keeps the total length of `tokens` constant throughout the replacement
                        // process, allowing us to do all replacements without adjusting indices.
                        let target_len = target.is_some() as usize;
                        tokens.splice(
                            (node_range.0.start as usize)..(node_range.0.end as usize),
                            target.into_iter().map(|target| FlatToken::AttrsTarget(target)).chain(
                                iter::repeat(FlatToken::Empty)
                                    .take(node_range.0.len() - target_len),
                            ),
                        );
                    }
                    make_attr_token_stream(tokens.into_iter(), *break_last_token)
                }
            }
        }
    }
}

/// A helper struct used when building an `AttrTokenStream` from
/// a `LazyAttrTokenStream`. Both delimiter and non-delimited tokens
/// are stored as `FlatToken::Token`. A vector of `FlatToken`s
/// is then 'parsed' to build up an `AttrTokenStream` with nested
/// `AttrTokenTree::Delimited` tokens.
#[derive(Debug, Clone)]
enum FlatToken {
    /// A token - this holds both delimiter (e.g. '{' and '}')
    /// and non-delimiter tokens
    Token((Token, Spacing)),
    /// Holds the `AttrsTarget` for an AST node. The `AttrsTarget` is inserted
    /// directly into the constructed `AttrTokenStream` as an
    /// `AttrTokenTree::AttrsTarget`.
    AttrsTarget(AttrsTarget),
    /// A special 'empty' token that is ignored during the conversion
    /// to an `AttrTokenStream`. This is used to simplify the
    /// handling of replace ranges.
    Empty,
}

/// An `AttrTokenStream` is similar to a `TokenStream`, but with extra
/// information about the tokens for attribute targets. This is used
/// during expansion to perform early cfg-expansion, and to process attributes
/// during proc-macro invocations.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct AttrTokenStream(pub Arc<Vec<AttrTokenTree>>);

/// Converts a flattened iterator of tokens (including open and close delimiter tokens) into an
/// `AttrTokenStream`, creating an `AttrTokenTree::Delimited` for each matching pair of open and
/// close delims.
fn make_attr_token_stream(
    iter: impl Iterator<Item = FlatToken>,
    break_last_token: u32,
) -> AttrTokenStream {
    #[derive(Debug)]
    struct FrameData {
        // This is `None` for the first frame, `Some` for all others.
        open_delim_sp: Option<(Delimiter, Span, Spacing)>,
        inner: Vec<AttrTokenTree>,
    }
    // The stack always has at least one element. Storing it separately makes for shorter code.
    let mut stack_top = FrameData { open_delim_sp: None, inner: vec![] };
    let mut stack_rest = vec![];
    for flat_token in iter {
        match flat_token {
            FlatToken::Token((token @ Token { kind, span }, spacing)) => {
                if let Some(delim) = kind.open_delim() {
                    stack_rest.push(mem::replace(
                        &mut stack_top,
                        FrameData { open_delim_sp: Some((delim, span, spacing)), inner: vec![] },
                    ));
                } else if let Some(delim) = kind.close_delim() {
                    let frame_data = mem::replace(&mut stack_top, stack_rest.pop().unwrap());
                    let (open_delim, open_sp, open_spacing) = frame_data.open_delim_sp.unwrap();
                    assert!(
                        open_delim.eq_ignoring_invisible_origin(&delim),
                        "Mismatched open/close delims: open={open_delim:?} close={span:?}"
                    );
                    let dspan = DelimSpan::from_pair(open_sp, span);
                    let dspacing = DelimSpacing::new(open_spacing, spacing);
                    let stream = AttrTokenStream::new(frame_data.inner);
                    let delimited = AttrTokenTree::Delimited(dspan, dspacing, delim, stream);
                    stack_top.inner.push(delimited);
                } else {
                    stack_top.inner.push(AttrTokenTree::Token(token, spacing))
                }
            }
            FlatToken::AttrsTarget(target) => {
                stack_top.inner.push(AttrTokenTree::AttrsTarget(target))
            }
            FlatToken::Empty => {}
        }
    }

    if break_last_token > 0 {
        let last_token = stack_top.inner.pop().unwrap();
        if let AttrTokenTree::Token(last_token, spacing) = last_token {
            let (unglued, _) = last_token.kind.break_two_token_op(break_last_token).unwrap();

            // Tokens are always ASCII chars, so we can use byte arithmetic here.
            let mut first_span = last_token.span.shrink_to_lo();
            first_span =
                first_span.with_hi(first_span.lo() + rustc_span::BytePos(break_last_token));

            stack_top.inner.push(AttrTokenTree::Token(Token::new(unglued, first_span), spacing));
        } else {
            panic!("Unexpected last token {last_token:?}")
        }
    }
    AttrTokenStream::new(stack_top.inner)
}

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

/// A `TokenStream` is an abstract sequence of tokens, organized into [`TokenTree`]s.
#[derive(Clone, Debug, Default, Encodable, Decodable)]
pub struct TokenStream(pub(crate) Arc<Vec<TokenTree>>);

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
                    ) if (token_left.is_non_reserved_ident() || token_left.is_lit())
                        && (token_right.is_non_reserved_ident() || token_right.is_lit()) =>
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

impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for TokenStream {}

impl FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(iter: I) -> Self {
        TokenStream::new(iter.into_iter().collect::<Vec<TokenTree>>())
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

#[derive(Clone, Debug)]
pub struct TokenTreeCursor {
    stream: TokenStream,
    /// Points to the current token tree in the stream. In `TokenCursor::curr`,
    /// this can be any token tree. In `TokenCursor::stack`, this is always a
    /// `TokenTree::Delimited`.
    index: usize,
}

impl TokenTreeCursor {
    #[inline]
    pub fn new(stream: TokenStream) -> Self {
        TokenTreeCursor { stream, index: 0 }
    }

    #[inline]
    pub fn curr(&self) -> Option<&TokenTree> {
        self.stream.get(self.index)
    }

    pub fn look_ahead(&self, n: usize) -> Option<&TokenTree> {
        self.stream.get(self.index + n)
    }

    #[inline]
    pub fn bump(&mut self) {
        self.index += 1;
    }

    // For skipping ahead in rare circumstances.
    #[inline]
    pub fn bump_to_end(&mut self) {
        self.index = self.stream.len();
    }
}

/// A `TokenStream` cursor that produces `Token`s. It's a bit odd that
/// we (a) lex tokens into a nice tree structure (`TokenStream`), and then (b)
/// use this type to emit them as a linear sequence. But a linear sequence is
/// what the parser expects, for the most part.
#[derive(Clone, Debug)]
pub struct TokenCursor {
    // Cursor for the current (innermost) token stream. The index within the
    // cursor can point to any token tree in the stream (or one past the end).
    // The delimiters for this token stream are found in `self.stack.last()`;
    // if that is `None` we are in the outermost token stream which never has
    // delimiters.
    pub curr: TokenTreeCursor,

    // Token streams surrounding the current one. The index within each cursor
    // always points to a `TokenTree::Delimited`.
    pub stack: Vec<TokenTreeCursor>,
}

impl TokenCursor {
    pub fn next(&mut self) -> (Token, Spacing) {
        self.inlined_next()
    }

    /// This always-inlined version should only be used on hot code paths.
    #[inline(always)]
    pub fn inlined_next(&mut self) -> (Token, Spacing) {
        loop {
            // FIXME: we currently don't return `Delimiter::Invisible` open/close delims. To fix
            // #67062 we will need to, whereupon the `delim != Delimiter::Invisible` conditions
            // below can be removed.
            if let Some(tree) = self.curr.curr() {
                match tree {
                    &TokenTree::Token(token, spacing) => {
                        debug_assert!(!token.kind.is_delim());
                        let res = (token, spacing);
                        self.curr.bump();
                        return res;
                    }
                    &TokenTree::Delimited(sp, spacing, delim, ref tts) => {
                        let trees = TokenTreeCursor::new(tts.clone());
                        self.stack.push(mem::replace(&mut self.curr, trees));
                        if !delim.skip() {
                            return (Token::new(delim.as_open_token_kind(), sp.open), spacing.open);
                        }
                        // No open delimiter to return; continue on to the next iteration.
                    }
                };
            } else if let Some(parent) = self.stack.pop() {
                // We have exhausted this token stream. Move back to its parent token stream.
                let Some(&TokenTree::Delimited(span, spacing, delim, _)) = parent.curr() else {
                    panic!("parent should be Delimited")
                };
                self.curr = parent;
                self.curr.bump(); // move past the `Delimited`
                if !delim.skip() {
                    return (Token::new(delim.as_close_token_kind(), span.close), spacing.close);
                }
                // No close delimiter to return; continue on to the next iteration.
            } else {
                // We have exhausted the outermost token stream. The use of
                // `Spacing::Alone` is arbitrary and immaterial, because the
                // `Eof` token's spacing is never used.
                return (Token::new(token::Eof, DUMMY_SP), Spacing::Alone);
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Encodable, Decodable, HashStable_Generic, Walkable)]
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
    static_assert_size!(LazyAttrTokenStreamInner, 88);
    static_assert_size!(Option<LazyAttrTokenStream>, 8); // must be small, used in many AST nodes
    static_assert_size!(TokenStream, 8);
    static_assert_size!(TokenTree, 32);
    // tidy-alphabetical-end
}
