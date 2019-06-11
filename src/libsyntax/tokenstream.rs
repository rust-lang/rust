//! # Token Streams
//!
//! `TokenStream`s represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence (eg stream) of `TokenTree`s,
//! which are themselves a single `Token` or a `Delimited` subsequence of tokens.
//!
//! ## Ownership
//!
//! `TokenStreams` are persistent data structures constructed as ropes with reference
//! counted-children. In general, this means that calling an operation on a `TokenStream`
//! (such as `slice`) produces an entirely new `TokenStream` from the borrowed reference to
//! the original. This essentially coerces `TokenStream`s into 'views' of their subparts,
//! and a borrowed `TokenStream` is sufficient to build an owned `TokenStream` without taking
//! ownership of the original.

use crate::ext::base;
use crate::ext::tt::{macro_parser, quoted};
use crate::parse::Directory;
use crate::parse::token::{self, DelimToken, Token, TokenKind};
use crate::print::pprust;

use syntax_pos::{BytePos, Mark, Span, DUMMY_SP};
#[cfg(target_arch = "x86_64")]
use rustc_data_structures::static_assert_size;
use rustc_data_structures::sync::Lrc;
use serialize::{Decoder, Decodable, Encoder, Encodable};
use smallvec::{SmallVec, smallvec};

use std::borrow::Cow;
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
#[derive(Debug, Clone, PartialEq, RustcEncodable, RustcDecodable)]
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
{}

impl TokenTree {
    /// Use this token tree as a matcher to parse given tts.
    pub fn parse(cx: &base::ExtCtxt<'_>, mtch: &[quoted::TokenTree], tts: TokenStream)
                 -> macro_parser::NamedParseResult {
        // `None` is because we're not interpolating
        let directory = Directory {
            path: Cow::from(cx.current_expansion.module.directory.as_path()),
            ownership: cx.current_expansion.directory_ownership,
        };
        macro_parser::parse(cx.parse_sess(), tts, mtch, Some(directory), true)
    }

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

    // See comments in `Nonterminal::to_tokenstream` for why we care about
    // *probably* equal here rather than actual equality
    //
    // This is otherwise the same as `eq_unspanned`, only recursing with a
    // different method.
    pub fn probably_equal_for_proc_macro(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (TokenTree::Token(token), TokenTree::Token(token2)) => {
                token.probably_equal_for_proc_macro(token2)
            }
            (TokenTree::Delimited(_, delim, tts), TokenTree::Delimited(_, delim2, tts2)) => {
                delim == delim2 && tts.probably_equal_for_proc_macro(&tts2)
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
        TokenStream::new(vec![(self, Joint)])
    }

    pub fn token(kind: TokenKind, span: Span) -> TokenTree {
        TokenTree::Token(Token::new(kind, span))
    }

    /// Returns the opening delimiter as a token tree.
    pub fn open_tt(span: Span, delim: DelimToken) -> TokenTree {
        let open_span = if span.is_dummy() {
            span
        } else {
            span.with_hi(span.lo() + BytePos(delim.len() as u32))
        };
        TokenTree::token(token::OpenDelim(delim), open_span)
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(span: Span, delim: DelimToken) -> TokenTree {
        let close_span = if span.is_dummy() {
            span
        } else {
            span.with_lo(span.hi() - BytePos(delim.len() as u32))
        };
        TokenTree::token(token::CloseDelim(delim), close_span)
    }
}

/// # Token Streams
///
/// A `TokenStream` is an abstract sequence of tokens, organized into `TokenTree`s.
/// The goal is for procedural macros to work with `TokenStream`s and `TokenTree`s
/// instead of a representation of the abstract syntax tree.
/// Today's `TokenTree`s can still contain AST via `token::Interpolated` for back-compat.
///
/// The use of `Option` is an optimization that avoids the need for an
/// allocation when the stream is empty. However, it is not guaranteed that an
/// empty stream is represented with `None`; it may be represented as a `Some`
/// around an empty `Vec`.
#[derive(Clone, Debug)]
pub struct TokenStream(pub Option<Lrc<Vec<TreeAndJoint>>>);

pub type TreeAndJoint = (TokenTree, IsJoint);

// `TokenStream` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(TokenStream, 8);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IsJoint {
    Joint,
    NonJoint
}

use IsJoint::*;

impl TokenStream {
    /// Given a `TokenStream` with a `Stream` of only two arguments, return a new `TokenStream`
    /// separating the two arguments with a comma for diagnostic suggestions.
    pub(crate) fn add_comma(&self) -> Option<(TokenStream, Span)> {
        // Used to suggest if a user writes `foo!(a b);`
        if let Some(ref stream) = self.0 {
            let mut suggestion = None;
            let mut iter = stream.iter().enumerate().peekable();
            while let Some((pos, ts)) = iter.next() {
                if let Some((_, next)) = iter.peek() {
                    let sp = match (&ts, &next) {
                        (_, (TokenTree::Token(Token { kind: token::Comma, .. }), _)) => continue,
                        ((TokenTree::Token(token_left), NonJoint),
                         (TokenTree::Token(token_right), _))
                        if ((token_left.is_ident() && !token_left.is_reserved_ident())
                            || token_left.is_lit()) &&
                            ((token_right.is_ident() && !token_right.is_reserved_ident())
                            || token_right.is_lit()) => token_left.span,
                        ((TokenTree::Delimited(sp, ..), NonJoint), _) => sp.entire(),
                        _ => continue,
                    };
                    let sp = sp.shrink_to_hi();
                    let comma = (TokenTree::token(token::Comma, sp), NonJoint);
                    suggestion = Some((pos, comma, sp));
                }
            }
            if let Some((pos, comma, sp)) = suggestion {
                let mut new_stream = vec![];
                let parts = stream.split_at(pos + 1);
                new_stream.extend_from_slice(parts.0);
                new_stream.push(comma);
                new_stream.extend_from_slice(parts.1);
                return Some((TokenStream::new(new_stream), sp));
            }
        }
        None
    }
}

impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream::new(vec![(tree, NonJoint)])
    }
}

impl From<TokenTree> for TreeAndJoint {
    fn from(tree: TokenTree) -> TreeAndJoint {
        (tree, NonJoint)
    }
}

impl<T: Into<TokenStream>> iter::FromIterator<T> for TokenStream {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        TokenStream::from_streams(iter.into_iter().map(Into::into).collect::<SmallVec<_>>())
    }
}

impl Eq for TokenStream {}

impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.trees().eq(other.trees())
    }
}

impl TokenStream {
    pub fn len(&self) -> usize {
        if let Some(ref slice) = self.0 {
            slice.len()
        } else {
            0
        }
    }

    pub fn empty() -> TokenStream {
        TokenStream(None)
    }

    pub fn is_empty(&self) -> bool {
        match self.0 {
            None => true,
            Some(ref stream) => stream.is_empty(),
        }
    }

    pub(crate) fn from_streams(mut streams: SmallVec<[TokenStream; 2]>) -> TokenStream {
        match streams.len() {
            0 => TokenStream::empty(),
            1 => streams.pop().unwrap(),
            _ => {
                // rust-lang/rust#57735: pre-allocate vector to avoid
                // quadratic blow-up due to on-the-fly reallocations.
                let tree_count = streams.iter()
                    .map(|ts| match &ts.0 { None => 0, Some(s) => s.len() })
                    .sum();
                let mut vec = Vec::with_capacity(tree_count);

                for stream in streams {
                    match stream.0 {
                        None => {},
                        Some(stream2) => vec.extend(stream2.iter().cloned()),
                    }
                }
                TokenStream::new(vec)
            }
        }
    }

    pub fn new(streams: Vec<TreeAndJoint>) -> TokenStream {
        match streams.len() {
            0 => TokenStream(None),
            _ => TokenStream(Some(Lrc::new(streams))),
        }
    }

    pub fn append_to_tree_and_joint_vec(self, vec: &mut Vec<TreeAndJoint>) {
        if let Some(stream) = self.0 {
            vec.extend(stream.iter().cloned());
        }
    }

    pub fn trees(&self) -> Cursor {
        self.clone().into_trees()
    }

    pub fn into_trees(self) -> Cursor {
        Cursor::new(self)
    }

    /// Compares two TokenStreams, checking equality without regarding span information.
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

    // See comments in `Nonterminal::to_tokenstream` for why we care about
    // *probably* equal here rather than actual equality
    //
    // This is otherwise the same as `eq_unspanned`, only recursing with a
    // different method.
    pub fn probably_equal_for_proc_macro(&self, other: &TokenStream) -> bool {
        // When checking for `probably_eq`, we ignore certain tokens that aren't
        // preserved in the AST. Because they are not preserved, the pretty
        // printer arbitrarily adds or removes them when printing as token
        // streams, making a comparison between a token stream generated from an
        // AST and a token stream which was parsed into an AST more reliable.
        fn semantic_tree(tree: &TokenTree) -> bool {
            if let TokenTree::Token(token) = tree {
                if let
                    // The pretty printer tends to add trailing commas to
                    // everything, and in particular, after struct fields.
                    | token::Comma
                    // The pretty printer emits `NoDelim` as whitespace.
                    | token::OpenDelim(DelimToken::NoDelim)
                    | token::CloseDelim(DelimToken::NoDelim)
                    // The pretty printer collapses many semicolons into one.
                    | token::Semi
                    // The pretty printer collapses whitespace arbitrarily and can
                    // introduce whitespace from `NoDelim`.
                    | token::Whitespace
                    // The pretty printer can turn `$crate` into `::crate_name`
                    | token::ModSep = token.kind {
                    return false;
                }
            }
            true
        }

        let mut t1 = self.trees().filter(semantic_tree);
        let mut t2 = other.trees().filter(semantic_tree);
        for (t1, t2) in t1.by_ref().zip(t2.by_ref()) {
            if !t1.probably_equal_for_proc_macro(&t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    pub fn map_enumerated<F: FnMut(usize, TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        TokenStream(self.0.map(|stream| {
            Lrc::new(
                stream
                    .iter()
                    .enumerate()
                    .map(|(i, (tree, is_joint))| (f(i, tree.clone()), *is_joint))
                    .collect())
        }))
    }

    pub fn map<F: FnMut(TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        TokenStream(self.0.map(|stream| {
            Lrc::new(
                stream
                    .iter()
                    .map(|(tree, is_joint)| (f(tree.clone()), *is_joint))
                    .collect())
        }))
    }

    fn first_tree_and_joint(&self) -> Option<TreeAndJoint> {
        self.0.as_ref().map(|stream| {
            stream.first().unwrap().clone()
        })
    }

    fn last_tree_if_joint(&self) -> Option<TokenTree> {
        match self.0 {
            None => None,
            Some(ref stream) => {
                if let (tree, Joint) = stream.last().unwrap() {
                    Some(tree.clone())
                } else {
                    None
                }
            }
        }
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
        let stream = stream.into();
        let last_tree_if_joint = self.0.last().and_then(TokenStream::last_tree_if_joint);
        if let Some(TokenTree::Token(last_token)) = last_tree_if_joint {
            if let Some((TokenTree::Token(token), is_joint)) = stream.first_tree_and_joint() {
                if let Some(glued_tok) = last_token.glue(token) {
                    let last_stream = self.0.pop().unwrap();
                    self.push_all_but_last_tree(&last_stream);
                    let glued_tt = TokenTree::Token(glued_tok);
                    let glued_tokenstream = TokenStream::new(vec![(glued_tt, is_joint)]);
                    self.0.push(glued_tokenstream);
                    self.push_all_but_first_tree(&stream);
                    return
                }
            }
        }
        self.0.push(stream);
    }

    pub fn build(self) -> TokenStream {
        TokenStream::from_streams(self.0)
    }

    fn push_all_but_last_tree(&mut self, stream: &TokenStream) {
        if let Some(ref streams) = stream.0 {
            let len = streams.len();
            match len {
                1 => {}
                _ => self.0.push(TokenStream(Some(Lrc::new(streams[0 .. len - 1].to_vec())))),
            }
        }
    }

    fn push_all_but_first_tree(&mut self, stream: &TokenStream) {
        if let Some(ref streams) = stream.0 {
            let len = streams.len();
            match len {
                1 => {}
                _ => self.0.push(TokenStream(Some(Lrc::new(streams[1 .. len].to_vec())))),
            }
        }
    }
}

#[derive(Clone)]
pub struct Cursor {
    pub stream: TokenStream,
    index: usize,
}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.next_with_joint().map(|(tree, _)| tree)
    }
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Cursor { stream, index: 0 }
    }

    pub fn next_with_joint(&mut self) -> Option<TreeAndJoint> {
        match self.stream.0 {
            None => None,
            Some(ref stream) => {
                if self.index < stream.len() {
                    self.index += 1;
                    Some(stream[self.index - 1].clone())
                } else {
                    None
                }
            }
        }
    }

    pub fn append(&mut self, new_stream: TokenStream) {
        if new_stream.is_empty() {
            return;
        }
        let index = self.index;
        let stream = mem::replace(&mut self.stream, TokenStream(None));
        *self = TokenStream::from_streams(smallvec![stream, new_stream]).into_trees();
        self.index = index;
    }

    pub fn look_ahead(&self, n: usize) -> Option<TokenTree> {
        match self.stream.0 {
            None => None,
            Some(ref stream) => stream[self.index ..].get(n).map(|(tree, _)| tree.clone()),
        }
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&pprust::tokens_to_string(self.clone()))
    }
}

impl Encodable for TokenStream {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        self.trees().collect::<Vec<_>>().encode(encoder)
    }
}

impl Decodable for TokenStream {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<TokenStream, D::Error> {
        Vec::<TokenTree>::decode(decoder).map(|vec| vec.into_iter().collect())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, RustcEncodable, RustcDecodable)]
pub struct DelimSpan {
    pub open: Span,
    pub close: Span,
}

impl DelimSpan {
    pub fn from_single(sp: Span) -> Self {
        DelimSpan {
            open: sp,
            close: sp,
        }
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

    pub fn apply_mark(self, mark: Mark) -> Self {
        DelimSpan {
            open: self.open.apply_mark(mark),
            close: self.close.apply_mark(mark),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Name;
    use crate::with_default_globals;
    use crate::util::parser_testing::string_to_stream;
    use syntax_pos::{Span, BytePos, NO_EXPANSION};

    fn string_to_ts(string: &str) -> TokenStream {
        string_to_stream(string.to_owned())
    }

    fn sp(a: u32, b: u32) -> Span {
        Span::new(BytePos(a), BytePos(b), NO_EXPANSION)
    }

    #[test]
    fn test_concat() {
        with_default_globals(|| {
            let test_res = string_to_ts("foo::bar::baz");
            let test_fst = string_to_ts("foo::bar");
            let test_snd = string_to_ts("::baz");
            let eq_res = TokenStream::from_streams(smallvec![test_fst, test_snd]);
            assert_eq!(test_res.trees().count(), 5);
            assert_eq!(eq_res.trees().count(), 5);
            assert_eq!(test_res.eq_unspanned(&eq_res), true);
        })
    }

    #[test]
    fn test_to_from_bijection() {
        with_default_globals(|| {
            let test_start = string_to_ts("foo::bar(baz)");
            let test_end = test_start.trees().collect();
            assert_eq!(test_start, test_end)
        })
    }

    #[test]
    fn test_eq_0() {
        with_default_globals(|| {
            let test_res = string_to_ts("foo");
            let test_eqs = string_to_ts("foo");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_eq_1() {
        with_default_globals(|| {
            let test_res = string_to_ts("::bar::baz");
            let test_eqs = string_to_ts("::bar::baz");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_eq_3() {
        with_default_globals(|| {
            let test_res = string_to_ts("");
            let test_eqs = string_to_ts("");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_diseq_0() {
        with_default_globals(|| {
            let test_res = string_to_ts("::bar::baz");
            let test_eqs = string_to_ts("bar::baz");
            assert_eq!(test_res == test_eqs, false)
        })
    }

    #[test]
    fn test_diseq_1() {
        with_default_globals(|| {
            let test_res = string_to_ts("(bar,baz)");
            let test_eqs = string_to_ts("bar,baz");
            assert_eq!(test_res == test_eqs, false)
        })
    }

    #[test]
    fn test_is_empty() {
        with_default_globals(|| {
            let test0: TokenStream = Vec::<TokenTree>::new().into_iter().collect();
            let test1: TokenStream =
                TokenTree::token(token::Ident(Name::intern("a"), false), sp(0, 1)).into();
            let test2 = string_to_ts("foo(bar::baz)");

            assert_eq!(test0.is_empty(), true);
            assert_eq!(test1.is_empty(), false);
            assert_eq!(test2.is_empty(), false);
        })
    }

    #[test]
    fn test_dotdotdot() {
        with_default_globals(|| {
            let mut builder = TokenStreamBuilder::new();
            builder.push(TokenTree::token(token::Dot, sp(0, 1)).joint());
            builder.push(TokenTree::token(token::Dot, sp(1, 2)).joint());
            builder.push(TokenTree::token(token::Dot, sp(2, 3)));
            let stream = builder.build();
            assert!(stream.eq_unspanned(&string_to_ts("...")));
            assert_eq!(stream.trees().count(), 1);
        })
    }
}
