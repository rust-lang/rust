// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! # Token Streams
//!
//! `TokenStream`s represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence (eg stream) of `TokenTree`s,
//! which are themselves a single `Token` or a `Delimited` subsequence of tokens.
//!
//! ## Ownership
//! `TokenStreams` are persistent data structures constructed as ropes with reference
//! counted-children. In general, this means that calling an operation on a `TokenStream`
//! (such as `slice`) produces an entirely new `TokenStream` from the borrowed reference to
//! the original. This essentially coerces `TokenStream`s into 'views' of their subparts,
//! and a borrowed `TokenStream` is sufficient to build an owned `TokenStream` without taking
//! ownership of the original.

use syntax_pos::{BytePos, Span, DUMMY_SP};
use ext::base;
use ext::tt::{macro_parser, quoted};
use parse::Directory;
use parse::token::{self, Token};
use print::pprust;
use serialize::{Decoder, Decodable, Encoder, Encodable};
use util::RcVec;

use std::borrow::Cow;
use std::{fmt, iter, mem};

/// A delimited sequence of token trees
#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug)]
pub struct Delimited {
    /// The type of delimiter
    pub delim: token::DelimToken,
    /// The delimited sequence of token trees
    pub tts: ThinTokenStream,
}

impl Delimited {
    /// Returns the opening delimiter as a token.
    pub fn open_token(&self) -> token::Token {
        token::OpenDelim(self.delim)
    }

    /// Returns the closing delimiter as a token.
    pub fn close_token(&self) -> token::Token {
        token::CloseDelim(self.delim)
    }

    /// Returns the opening delimiter as a token tree.
    pub fn open_tt(&self, span: Span) -> TokenTree {
        let open_span = if span.is_dummy() {
            span
        } else {
            span.with_hi(span.lo() + BytePos(self.delim.len() as u32))
        };
        TokenTree::Token(open_span, self.open_token())
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(&self, span: Span) -> TokenTree {
        let close_span = if span.is_dummy() {
            span
        } else {
            span.with_lo(span.hi() - BytePos(self.delim.len() as u32))
        };
        TokenTree::Token(close_span, self.close_token())
    }

    /// Returns the token trees inside the delimiters.
    pub fn stream(&self) -> TokenStream {
        self.tts.clone().into()
    }
}

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
    Token(Span, token::Token),
    /// A delimited sequence of token trees
    Delimited(Span, Delimited),
}

impl TokenTree {
    /// Use this token tree as a matcher to parse given tts.
    pub fn parse(cx: &base::ExtCtxt, mtch: &[quoted::TokenTree], tts: TokenStream)
                 -> macro_parser::NamedParseResult {
        // `None` is because we're not interpolating
        let directory = Directory {
            path: Cow::from(cx.current_expansion.module.directory.as_path()),
            ownership: cx.current_expansion.directory_ownership,
        };
        macro_parser::parse(cx.parse_sess(), tts, mtch, Some(directory), true)
    }

    /// Check if this TokenTree is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => tk == tk2,
            (&TokenTree::Delimited(_, ref dl), &TokenTree::Delimited(_, ref dl2)) => {
                dl.delim == dl2.delim &&
                dl.stream().eq_unspanned(&dl2.stream())
            }
            (_, _) => false,
        }
    }

    // See comments in `interpolated_to_tokenstream` for why we care about
    // *probably* equal here rather than actual equality
    //
    // This is otherwise the same as `eq_unspanned`, only recursing with a
    // different method.
    pub fn probably_equal_for_proc_macro(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => {
                tk.probably_equal_for_proc_macro(tk2)
            }
            (&TokenTree::Delimited(_, ref dl), &TokenTree::Delimited(_, ref dl2)) => {
                dl.delim == dl2.delim &&
                dl.stream().probably_equal_for_proc_macro(&dl2.stream())
            }
            (_, _) => false,
        }
    }

    /// Retrieve the TokenTree's span.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Token(sp, _) | TokenTree::Delimited(sp, _) => sp,
        }
    }

    /// Modify the `TokenTree`'s span inplace.
    pub fn set_span(&mut self, span: Span) {
        match *self {
            TokenTree::Token(ref mut sp, _) | TokenTree::Delimited(ref mut sp, _) => {
                *sp = span;
            }
        }
    }

    /// Indicates if the stream is a token that is equal to the provided token.
    pub fn eq_token(&self, t: Token) -> bool {
        match *self {
            TokenTree::Token(_, ref tk) => *tk == t,
            _ => false,
        }
    }

    pub fn joint(self) -> TokenStream {
        TokenStream { kind: TokenStreamKind::JointTree(self) }
    }
}

/// # Token Streams
///
/// A `TokenStream` is an abstract sequence of tokens, organized into `TokenTree`s.
/// The goal is for procedural macros to work with `TokenStream`s and `TokenTree`s
/// instead of a representation of the abstract syntax tree.
/// Today's `TokenTree`s can still contain AST via `Token::Interpolated` for back-compat.
#[derive(Clone, Debug)]
pub struct TokenStream {
    kind: TokenStreamKind,
}

impl TokenStream {
    /// Given a `TokenStream` with a `Stream` of only two arguments, return a new `TokenStream`
    /// separating the two arguments with a comma for diagnostic suggestions.
    pub(crate) fn add_comma(&self) -> Option<(TokenStream, Span)> {
        // Used to suggest if a user writes `foo!(a b);`
        if let TokenStreamKind::Stream(ref slice) = self.kind {
            let mut suggestion = None;
            let mut iter = slice.iter().enumerate().peekable();
            while let Some((pos, ts)) = iter.next() {
                if let Some((_, next)) = iter.peek() {
                    match (ts, next) {
                        (TokenStream {
                            kind: TokenStreamKind::Tree(TokenTree::Token(_, token::Token::Comma))
                        }, _) |
                        (_, TokenStream {
                            kind: TokenStreamKind::Tree(TokenTree::Token(_, token::Token::Comma))
                        }) => {}
                        (TokenStream {
                            kind: TokenStreamKind::Tree(TokenTree::Token(sp, _))
                        }, _) |
                        (TokenStream {
                            kind: TokenStreamKind::Tree(TokenTree::Delimited(sp, _))
                        }, _) => {
                            let sp = sp.shrink_to_hi();
                            let comma = TokenStream {
                                kind: TokenStreamKind::Tree(TokenTree::Token(sp, token::Comma)),
                            };
                            suggestion = Some((pos, comma, sp));
                        }
                        _ => {}
                    }
                }
            }
            if let Some((pos, comma, sp)) = suggestion {
                let mut new_slice = vec![];
                let parts = slice.split_at(pos + 1);
                new_slice.extend_from_slice(parts.0);
                new_slice.push(comma);
                new_slice.extend_from_slice(parts.1);
                let slice = RcVec::new(new_slice);
                return Some((TokenStream { kind: TokenStreamKind::Stream(slice) }, sp));
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
enum TokenStreamKind {
    Empty,
    Tree(TokenTree),
    JointTree(TokenTree),
    Stream(RcVec<TokenStream>),
}

impl From<TokenTree> for TokenStream {
    fn from(tt: TokenTree) -> TokenStream {
        TokenStream { kind: TokenStreamKind::Tree(tt) }
    }
}

impl From<Token> for TokenStream {
    fn from(token: Token) -> TokenStream {
        TokenTree::Token(DUMMY_SP, token).into()
    }
}

impl<T: Into<TokenStream>> iter::FromIterator<T> for TokenStream {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        TokenStream::concat(iter.into_iter().map(Into::into).collect::<Vec<_>>())
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let kind = mem::replace(&mut self.kind, TokenStreamKind::Empty);

        // Vector of token streams originally in self.
        let tts: Vec<TokenStream> = match kind {
            TokenStreamKind::Empty => {
                let mut vec = Vec::new();
                vec.reserve(iter.size_hint().0);
                vec
            }
            TokenStreamKind::Tree(_) | TokenStreamKind::JointTree(_) => {
                let mut vec = Vec::new();
                vec.reserve(1 + iter.size_hint().0);
                vec.push(TokenStream { kind });
                vec
            }
            TokenStreamKind::Stream(rc_vec) => match RcVec::try_unwrap(rc_vec) {
                Ok(mut vec) => {
                    // Extend in place using the existing capacity if possible.
                    // This is the fast path for libraries like `quote` that
                    // build a token stream.
                    vec.reserve(iter.size_hint().0);
                    vec
                }
                Err(rc_vec) => {
                    // Self is shared so we need to copy and extend that.
                    let mut vec = Vec::new();
                    vec.reserve(rc_vec.len() + iter.size_hint().0);
                    vec.extend_from_slice(&rc_vec);
                    vec
                }
            }
        };

        // Perform the extend, joining tokens as needed along the way.
        let mut builder = TokenStreamBuilder(tts);
        for stream in iter {
            builder.push(stream);
        }

        // Build the resulting token stream. If it contains more than one token,
        // preserve capacity in the vector in anticipation of the caller
        // performing additional calls to extend.
        let mut tts = builder.0;
        *self = match tts.len() {
            0 => TokenStream::empty(),
            1 => tts.pop().unwrap(),
            _ => TokenStream::concat_rc_vec(RcVec::new_preserving_capacity(tts)),
        };
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
        if let TokenStreamKind::Stream(ref slice) = self.kind {
            slice.len()
        } else {
            0
        }
    }

    pub fn empty() -> TokenStream {
        TokenStream { kind: TokenStreamKind::Empty }
    }

    pub fn is_empty(&self) -> bool {
        match self.kind {
            TokenStreamKind::Empty => true,
            _ => false,
        }
    }

    pub fn concat(mut streams: Vec<TokenStream>) -> TokenStream {
        match streams.len() {
            0 => TokenStream::empty(),
            1 => streams.pop().unwrap(),
            _ => TokenStream::concat_rc_vec(RcVec::new(streams)),
        }
    }

    fn concat_rc_vec(streams: RcVec<TokenStream>) -> TokenStream {
        TokenStream { kind: TokenStreamKind::Stream(streams) }
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

    // See comments in `interpolated_to_tokenstream` for why we care about
    // *probably* equal here rather than actual equality
    //
    // This is otherwise the same as `eq_unspanned`, only recursing with a
    // different method.
    pub fn probably_equal_for_proc_macro(&self, other: &TokenStream) -> bool {
        let mut t1 = self.trees();
        let mut t2 = other.trees();
        for (t1, t2) in t1.by_ref().zip(t2.by_ref()) {
            if !t1.probably_equal_for_proc_macro(&t2) {
                return false;
            }
        }
        t1.next().is_none() && t2.next().is_none()
    }

    /// Precondition: `self` consists of a single token tree.
    /// Returns true if the token tree is a joint operation w.r.t. `proc_macro::TokenNode`.
    pub fn as_tree(self) -> (TokenTree, bool /* joint? */) {
        match self.kind {
            TokenStreamKind::Tree(tree) => (tree, false),
            TokenStreamKind::JointTree(tree) => (tree, true),
            _ => unreachable!(),
        }
    }

    pub fn map_enumerated<F: FnMut(usize, TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        let mut trees = self.into_trees();
        let mut result = Vec::new();
        let mut i = 0;
        while let Some(stream) = trees.next_as_stream() {
            result.push(match stream.kind {
                TokenStreamKind::Tree(tree) => f(i, tree).into(),
                TokenStreamKind::JointTree(tree) => f(i, tree).joint(),
                _ => unreachable!()
            });
            i += 1;
        }
        TokenStream::concat(result)
    }

    pub fn map<F: FnMut(TokenTree) -> TokenTree>(self, mut f: F) -> TokenStream {
        let mut trees = self.into_trees();
        let mut result = Vec::new();
        while let Some(stream) = trees.next_as_stream() {
            result.push(match stream.kind {
                TokenStreamKind::Tree(tree) => f(tree).into(),
                TokenStreamKind::JointTree(tree) => f(tree).joint(),
                _ => unreachable!()
            });
        }
        TokenStream::concat(result)
    }

    fn first_tree_and_joint(&self) -> Option<(TokenTree, bool)> {
        match self.kind {
            TokenStreamKind::Empty => None,
            TokenStreamKind::Tree(ref tree) => Some((tree.clone(), false)),
            TokenStreamKind::JointTree(ref tree) => Some((tree.clone(), true)),
            TokenStreamKind::Stream(ref stream) => stream.first().unwrap().first_tree_and_joint(),
        }
    }

    fn last_tree_if_joint(&self) -> Option<TokenTree> {
        match self.kind {
            TokenStreamKind::Empty | TokenStreamKind::Tree(..) => None,
            TokenStreamKind::JointTree(ref tree) => Some(tree.clone()),
            TokenStreamKind::Stream(ref stream) => stream.last().unwrap().last_tree_if_joint(),
        }
    }
}

#[derive(Clone)]
pub struct TokenStreamBuilder(Vec<TokenStream>);

impl TokenStreamBuilder {
    pub fn new() -> TokenStreamBuilder {
        TokenStreamBuilder(Vec::new())
    }

    pub fn push<T: Into<TokenStream>>(&mut self, stream: T) {
        let stream = stream.into();
        let last_tree_if_joint = self.0.last().and_then(TokenStream::last_tree_if_joint);
        if let Some(TokenTree::Token(last_span, last_tok)) = last_tree_if_joint {
            if let Some((TokenTree::Token(span, tok), is_joint)) = stream.first_tree_and_joint() {
                if let Some(glued_tok) = last_tok.glue(tok) {
                    let last_stream = self.0.pop().unwrap();
                    self.push_all_but_last_tree(&last_stream);
                    let glued_span = last_span.to(span);
                    let glued_tt = TokenTree::Token(glued_span, glued_tok);
                    let glued_tokenstream = if is_joint {
                        glued_tt.joint()
                    } else {
                        glued_tt.into()
                    };
                    self.0.push(glued_tokenstream);
                    self.push_all_but_first_tree(&stream);
                    return
                }
            }
        }
        self.0.push(stream);
    }

    pub fn add<T: Into<TokenStream>>(mut self, stream: T) -> Self {
        self.push(stream);
        self
    }

    pub fn build(self) -> TokenStream {
        TokenStream::concat(self.0)
    }

    fn push_all_but_last_tree(&mut self, stream: &TokenStream) {
        if let TokenStreamKind::Stream(ref streams) = stream.kind {
            let len = streams.len();
            match len {
                1 => {}
                2 => self.0.push(streams[0].clone().into()),
                _ => self.0.push(TokenStream::concat_rc_vec(streams.sub_slice(0 .. len - 1))),
            }
            self.push_all_but_last_tree(&streams[len - 1])
        }
    }

    fn push_all_but_first_tree(&mut self, stream: &TokenStream) {
        if let TokenStreamKind::Stream(ref streams) = stream.kind {
            let len = streams.len();
            match len {
                1 => {}
                2 => self.0.push(streams[1].clone().into()),
                _ => self.0.push(TokenStream::concat_rc_vec(streams.sub_slice(1 .. len))),
            }
            self.push_all_but_first_tree(&streams[0])
        }
    }
}

#[derive(Clone)]
pub struct Cursor(CursorKind);

#[derive(Clone)]
enum CursorKind {
    Empty,
    Tree(TokenTree, bool /* consumed? */),
    JointTree(TokenTree, bool /* consumed? */),
    Stream(StreamCursor),
}

#[derive(Clone)]
struct StreamCursor {
    stream: RcVec<TokenStream>,
    index: usize,
    stack: Vec<(RcVec<TokenStream>, usize)>,
}

impl StreamCursor {
    fn new(stream: RcVec<TokenStream>) -> Self {
        StreamCursor { stream: stream, index: 0, stack: Vec::new() }
    }

    fn next_as_stream(&mut self) -> Option<TokenStream> {
        loop {
            if self.index < self.stream.len() {
                self.index += 1;
                let next = self.stream[self.index - 1].clone();
                match next.kind {
                    TokenStreamKind::Tree(..) | TokenStreamKind::JointTree(..) => return Some(next),
                    TokenStreamKind::Stream(stream) => self.insert(stream),
                    TokenStreamKind::Empty => {}
                }
            } else if let Some((stream, index)) = self.stack.pop() {
                self.stream = stream;
                self.index = index;
            } else {
                return None;
            }
        }
    }

    fn insert(&mut self, stream: RcVec<TokenStream>) {
        self.stack.push((mem::replace(&mut self.stream, stream),
                         mem::replace(&mut self.index, 0)));
    }
}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        self.next_as_stream().map(|stream| match stream.kind {
            TokenStreamKind::Tree(tree) | TokenStreamKind::JointTree(tree) => tree,
            _ => unreachable!()
        })
    }
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Cursor(match stream.kind {
            TokenStreamKind::Empty => CursorKind::Empty,
            TokenStreamKind::Tree(tree) => CursorKind::Tree(tree, false),
            TokenStreamKind::JointTree(tree) => CursorKind::JointTree(tree, false),
            TokenStreamKind::Stream(stream) => CursorKind::Stream(StreamCursor::new(stream)),
        })
    }

    pub fn next_as_stream(&mut self) -> Option<TokenStream> {
        let (stream, consumed) = match self.0 {
            CursorKind::Tree(ref tree, ref mut consumed @ false) =>
                (tree.clone().into(), consumed),
            CursorKind::JointTree(ref tree, ref mut consumed @ false) =>
                (tree.clone().joint(), consumed),
            CursorKind::Stream(ref mut cursor) => return cursor.next_as_stream(),
            _ => return None,
        };

        *consumed = true;
        Some(stream)
    }

    pub fn insert(&mut self, stream: TokenStream) {
        match self.0 {
            _ if stream.is_empty() => return,
            CursorKind::Empty => *self = stream.trees(),
            CursorKind::Tree(_, consumed) | CursorKind::JointTree(_, consumed) => {
                *self = TokenStream::concat(vec![self.original_stream(), stream]).trees();
                if consumed {
                    self.next();
                }
            }
            CursorKind::Stream(ref mut cursor) => {
                cursor.insert(ThinTokenStream::from(stream).0.unwrap());
            }
        }
    }

    pub fn original_stream(&self) -> TokenStream {
        match self.0 {
            CursorKind::Empty => TokenStream::empty(),
            CursorKind::Tree(ref tree, _) => tree.clone().into(),
            CursorKind::JointTree(ref tree, _) => tree.clone().joint(),
            CursorKind::Stream(ref cursor) => TokenStream::concat_rc_vec({
                cursor.stack.get(0).cloned().map(|(stream, _)| stream)
                    .unwrap_or(cursor.stream.clone())
            }),
        }
    }

    pub fn look_ahead(&self, n: usize) -> Option<TokenTree> {
        fn look_ahead(streams: &[TokenStream], mut n: usize) -> Result<TokenTree, usize> {
            for stream in streams {
                n = match stream.kind {
                    TokenStreamKind::Tree(ref tree) | TokenStreamKind::JointTree(ref tree)
                        if n == 0 => return Ok(tree.clone()),
                    TokenStreamKind::Tree(..) | TokenStreamKind::JointTree(..) => n - 1,
                    TokenStreamKind::Stream(ref stream) => match look_ahead(stream, n) {
                        Ok(tree) => return Ok(tree),
                        Err(n) => n,
                    },
                    _ => n,
                };
            }
            Err(n)
        }

        match self.0 {
            CursorKind::Empty |
            CursorKind::Tree(_, true) |
            CursorKind::JointTree(_, true) => Err(n),
            CursorKind::Tree(ref tree, false) |
            CursorKind::JointTree(ref tree, false) => look_ahead(&[tree.clone().into()], n),
            CursorKind::Stream(ref cursor) => {
                look_ahead(&cursor.stream[cursor.index ..], n).or_else(|mut n| {
                    for &(ref stream, index) in cursor.stack.iter().rev() {
                        n = match look_ahead(&stream[index..], n) {
                            Ok(tree) => return Ok(tree),
                            Err(n) => n,
                        }
                    }

                    Err(n)
                })
            }
        }.ok()
    }
}

/// The `TokenStream` type is large enough to represent a single `TokenTree` without allocation.
/// `ThinTokenStream` is smaller, but needs to allocate to represent a single `TokenTree`.
/// We must use `ThinTokenStream` in `TokenTree::Delimited` to avoid infinite size due to recursion.
#[derive(Debug, Clone)]
pub struct ThinTokenStream(Option<RcVec<TokenStream>>);

impl From<TokenStream> for ThinTokenStream {
    fn from(stream: TokenStream) -> ThinTokenStream {
        ThinTokenStream(match stream.kind {
            TokenStreamKind::Empty => None,
            TokenStreamKind::Tree(tree) => Some(RcVec::new(vec![tree.into()])),
            TokenStreamKind::JointTree(tree) => Some(RcVec::new(vec![tree.joint()])),
            TokenStreamKind::Stream(stream) => Some(stream),
        })
    }
}

impl From<ThinTokenStream> for TokenStream {
    fn from(stream: ThinTokenStream) -> TokenStream {
        stream.0.map(TokenStream::concat_rc_vec).unwrap_or_else(TokenStream::empty)
    }
}

impl Eq for ThinTokenStream {}

impl PartialEq<ThinTokenStream> for ThinTokenStream {
    fn eq(&self, other: &ThinTokenStream) -> bool {
        TokenStream::from(self.clone()) == TokenStream::from(other.clone())
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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

impl Encodable for ThinTokenStream {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
        TokenStream::from(self.clone()).encode(encoder)
    }
}

impl Decodable for ThinTokenStream {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<ThinTokenStream, D::Error> {
        TokenStream::decode(decoder).map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syntax::ast::Ident;
    use with_globals;
    use syntax_pos::{Span, BytePos, NO_EXPANSION};
    use parse::token::Token;
    use util::parser_testing::string_to_stream;

    fn string_to_ts(string: &str) -> TokenStream {
        string_to_stream(string.to_owned())
    }

    fn sp(a: u32, b: u32) -> Span {
        Span::new(BytePos(a), BytePos(b), NO_EXPANSION)
    }

    #[test]
    fn test_concat() {
        with_globals(|| {
            let test_res = string_to_ts("foo::bar::baz");
            let test_fst = string_to_ts("foo::bar");
            let test_snd = string_to_ts("::baz");
            let eq_res = TokenStream::concat(vec![test_fst, test_snd]);
            assert_eq!(test_res.trees().count(), 5);
            assert_eq!(eq_res.trees().count(), 5);
            assert_eq!(test_res.eq_unspanned(&eq_res), true);
        })
    }

    #[test]
    fn test_to_from_bijection() {
        with_globals(|| {
            let test_start = string_to_ts("foo::bar(baz)");
            let test_end = test_start.trees().collect();
            assert_eq!(test_start, test_end)
        })
    }

    #[test]
    fn test_eq_0() {
        with_globals(|| {
            let test_res = string_to_ts("foo");
            let test_eqs = string_to_ts("foo");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_eq_1() {
        with_globals(|| {
            let test_res = string_to_ts("::bar::baz");
            let test_eqs = string_to_ts("::bar::baz");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_eq_3() {
        with_globals(|| {
            let test_res = string_to_ts("");
            let test_eqs = string_to_ts("");
            assert_eq!(test_res, test_eqs)
        })
    }

    #[test]
    fn test_diseq_0() {
        with_globals(|| {
            let test_res = string_to_ts("::bar::baz");
            let test_eqs = string_to_ts("bar::baz");
            assert_eq!(test_res == test_eqs, false)
        })
    }

    #[test]
    fn test_diseq_1() {
        with_globals(|| {
            let test_res = string_to_ts("(bar,baz)");
            let test_eqs = string_to_ts("bar,baz");
            assert_eq!(test_res == test_eqs, false)
        })
    }

    #[test]
    fn test_is_empty() {
        with_globals(|| {
            let test0: TokenStream = Vec::<TokenTree>::new().into_iter().collect();
            let test1: TokenStream =
                TokenTree::Token(sp(0, 1), Token::Ident(Ident::from_str("a"), false)).into();
            let test2 = string_to_ts("foo(bar::baz)");

            assert_eq!(test0.is_empty(), true);
            assert_eq!(test1.is_empty(), false);
            assert_eq!(test2.is_empty(), false);
        })
    }

    #[test]
    fn test_dotdotdot() {
        let mut builder = TokenStreamBuilder::new();
        builder.push(TokenTree::Token(sp(0, 1), Token::Dot).joint());
        builder.push(TokenTree::Token(sp(1, 2), Token::Dot).joint());
        builder.push(TokenTree::Token(sp(2, 3), Token::Dot));
        let stream = builder.build();
        assert!(stream.eq_unspanned(&string_to_ts("...")));
        assert_eq!(stream.trees().count(), 1);
    }

    #[test]
    fn test_extend_empty() {
        with_globals(|| {
            // Append a token onto an empty token stream.
            let mut stream = TokenStream::empty();
            stream.extend(vec![string_to_ts("t")]);

            let expected = string_to_ts("t");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_nothing() {
        with_globals(|| {
            // Append nothing onto a token stream containing one token.
            let mut stream = string_to_ts("t");
            stream.extend(vec![]);

            let expected = string_to_ts("t");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_single() {
        with_globals(|| {
            // Append a token onto token stream containing a single token.
            let mut stream = string_to_ts("t1");
            stream.extend(vec![string_to_ts("t2")]);

            let expected = string_to_ts("t1 t2");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_in_place() {
        with_globals(|| {
            // Append a token onto token stream containing a reference counted
            // vec of tokens. The token stream has a reference count of 1 so
            // this can happen in place.
            let mut stream = string_to_ts("t1 t2");
            stream.extend(vec![string_to_ts("t3")]);

            let expected = string_to_ts("t1 t2 t3");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_copy() {
        with_globals(|| {
            // Append a token onto token stream containing a reference counted
            // vec of tokens. The token stream is shared so the extend takes
            // place on a copy.
            let mut stream = string_to_ts("t1 t2");
            let _incref = stream.clone();
            stream.extend(vec![string_to_ts("t3")]);

            let expected = string_to_ts("t1 t2 t3");
            assert!(stream.eq_unspanned(&expected));
        });
    }

    #[test]
    fn test_extend_no_join() {
        with_globals(|| {
            let first = TokenTree::Token(DUMMY_SP, Token::Dot);
            let second = TokenTree::Token(DUMMY_SP, Token::Dot);

            // Append a dot onto a token stream containing a dot, but do not
            // join them.
            let mut stream = TokenStream::from(first);
            stream.extend(vec![TokenStream::from(second)]);

            let expected = string_to_ts(". .");
            assert!(stream.eq_unspanned(&expected));

            let unexpected = string_to_ts("..");
            assert!(!stream.eq_unspanned(&unexpected));
        });
    }

    #[test]
    fn test_extend_join() {
        with_globals(|| {
            let first = TokenTree::Token(DUMMY_SP, Token::Dot).joint();
            let second = TokenTree::Token(DUMMY_SP, Token::Dot);

            // Append a dot onto a token stream containing a dot, forming a
            // dotdot.
            let mut stream = first;
            stream.extend(vec![TokenStream::from(second)]);

            let expected = string_to_ts("..");
            assert!(stream.eq_unspanned(&expected));

            let unexpected = string_to_ts(". .");
            assert!(!stream.eq_unspanned(&unexpected));
        });
    }
}
