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
use util::RcSlice;

use std::{fmt, iter, mem};
use std::hash::{self, Hash};

/// A delimited sequence of token trees
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
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
        let open_span = if span == DUMMY_SP {
            DUMMY_SP
        } else {
            Span { hi: span.lo + BytePos(self.delim.len() as u32), ..span }
        };
        TokenTree::Token(open_span, self.open_token())
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(&self, span: Span) -> TokenTree {
        let close_span = if span == DUMMY_SP {
            DUMMY_SP
        } else {
            Span { lo: span.hi - BytePos(self.delim.len() as u32), ..span }
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
#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
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
            path: cx.current_expansion.module.directory.clone(),
            ownership: cx.current_expansion.directory_ownership,
        };
        macro_parser::parse(cx.parse_sess(), tts, mtch, Some(directory))
    }

    /// Check if this TokenTree is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => tk == tk2,
            (&TokenTree::Delimited(_, ref dl), &TokenTree::Delimited(_, ref dl2)) => {
                dl.delim == dl2.delim &&
                dl.stream().trees().zip(dl2.stream().trees()).all(|(tt, tt2)| tt.eq_unspanned(&tt2))
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

    /// Indicates if the stream is a token that is equal to the provided token.
    pub fn eq_token(&self, t: Token) -> bool {
        match *self {
            TokenTree::Token(_, ref tk) => *tk == t,
            _ => false,
        }
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

#[derive(Clone, Debug)]
enum TokenStreamKind {
    Empty,
    Tree(TokenTree),
    Stream(RcSlice<TokenStream>),
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

impl Eq for TokenStream {}

impl PartialEq<TokenStream> for TokenStream {
    fn eq(&self, other: &TokenStream) -> bool {
        self.trees().eq(other.trees())
    }
}

impl TokenStream {
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
            1 => TokenStream::from(streams.pop().unwrap()),
            _ => TokenStream::concat_rc_slice(RcSlice::new(streams)),
        }
    }

    fn concat_rc_slice(streams: RcSlice<TokenStream>) -> TokenStream {
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
        for (t1, t2) in self.trees().zip(other.trees()) {
            if !t1.eq_unspanned(&t2) {
                return false;
            }
        }
        true
    }
}

pub struct Cursor(CursorKind);

enum CursorKind {
    Empty,
    Tree(TokenTree, bool /* consumed? */),
    Stream(StreamCursor),
}

struct StreamCursor {
    stream: RcSlice<TokenStream>,
    index: usize,
    stack: Vec<(RcSlice<TokenStream>, usize)>,
}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        let cursor = match self.0 {
            CursorKind::Stream(ref mut cursor) => cursor,
            CursorKind::Tree(ref tree, ref mut consumed @ false) => {
                *consumed = true;
                return Some(tree.clone());
            }
            _ => return None,
        };

        loop {
            if cursor.index < cursor.stream.len() {
                match cursor.stream[cursor.index].kind.clone() {
                    TokenStreamKind::Tree(tree) => {
                        cursor.index += 1;
                        return Some(tree);
                    }
                    TokenStreamKind::Stream(stream) => {
                        cursor.stack.push((mem::replace(&mut cursor.stream, stream),
                                           mem::replace(&mut cursor.index, 0) + 1));
                    }
                    TokenStreamKind::Empty => {
                        cursor.index += 1;
                    }
                }
            } else if let Some((stream, index)) = cursor.stack.pop() {
                cursor.stream = stream;
                cursor.index = index;
            } else {
                return None;
            }
        }
    }
}

impl Cursor {
    fn new(stream: TokenStream) -> Self {
        Cursor(match stream.kind {
            TokenStreamKind::Empty => CursorKind::Empty,
            TokenStreamKind::Tree(tree) => CursorKind::Tree(tree, false),
            TokenStreamKind::Stream(stream) => {
                CursorKind::Stream(StreamCursor { stream: stream, index: 0, stack: Vec::new() })
            }
        })
    }

    pub fn original_stream(self) -> TokenStream {
        match self.0 {
            CursorKind::Empty => TokenStream::empty(),
            CursorKind::Tree(tree, _) => tree.into(),
            CursorKind::Stream(cursor) => TokenStream::concat_rc_slice({
                cursor.stack.get(0).cloned().map(|(stream, _)| stream).unwrap_or(cursor.stream)
            }),
        }
    }

    pub fn look_ahead(&self, n: usize) -> Option<TokenTree> {
        fn look_ahead(streams: &[TokenStream], mut n: usize) -> Result<TokenTree, usize> {
            for stream in streams {
                n = match stream.kind {
                    TokenStreamKind::Tree(ref tree) if n == 0 => return Ok(tree.clone()),
                    TokenStreamKind::Tree(..) => n - 1,
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
            CursorKind::Empty | CursorKind::Tree(_, true) => Err(n),
            CursorKind::Tree(ref tree, false) => look_ahead(&[tree.clone().into()], n),
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
pub struct ThinTokenStream(Option<RcSlice<TokenStream>>);

impl From<TokenStream> for ThinTokenStream {
    fn from(stream: TokenStream) -> ThinTokenStream {
        ThinTokenStream(match stream.kind {
            TokenStreamKind::Empty => None,
            TokenStreamKind::Tree(tree) => Some(RcSlice::new(vec![tree.into()])),
            TokenStreamKind::Stream(stream) => Some(stream),
        })
    }
}

impl From<ThinTokenStream> for TokenStream {
    fn from(stream: ThinTokenStream) -> TokenStream {
        stream.0.map(TokenStream::concat_rc_slice).unwrap_or_else(TokenStream::empty)
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

impl Hash for TokenStream {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        for tree in self.trees() {
            tree.hash(state);
        }
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

impl Hash for ThinTokenStream {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        TokenStream::from(self.clone()).hash(state);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use syntax::ast::Ident;
    use syntax_pos::{Span, BytePos, NO_EXPANSION};
    use parse::token::Token;
    use util::parser_testing::string_to_stream;

    fn string_to_ts(string: &str) -> TokenStream {
        string_to_stream(string.to_owned())
    }

    fn sp(a: u32, b: u32) -> Span {
        Span {
            lo: BytePos(a),
            hi: BytePos(b),
            ctxt: NO_EXPANSION,
        }
    }

    #[test]
    fn test_concat() {
        let test_res = string_to_ts("foo::bar::baz");
        let test_fst = string_to_ts("foo::bar");
        let test_snd = string_to_ts("::baz");
        let eq_res = TokenStream::concat(vec![test_fst, test_snd]);
        assert_eq!(test_res.trees().count(), 5);
        assert_eq!(eq_res.trees().count(), 5);
        assert_eq!(test_res.eq_unspanned(&eq_res), true);
    }

    #[test]
    fn test_to_from_bijection() {
        let test_start = string_to_ts("foo::bar(baz)");
        let test_end = test_start.trees().collect();
        assert_eq!(test_start, test_end)
    }

    #[test]
    fn test_eq_0() {
        let test_res = string_to_ts("foo");
        let test_eqs = string_to_ts("foo");
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_eq_1() {
        let test_res = string_to_ts("::bar::baz");
        let test_eqs = string_to_ts("::bar::baz");
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_eq_3() {
        let test_res = string_to_ts("");
        let test_eqs = string_to_ts("");
        assert_eq!(test_res, test_eqs)
    }

    #[test]
    fn test_diseq_0() {
        let test_res = string_to_ts("::bar::baz");
        let test_eqs = string_to_ts("bar::baz");
        assert_eq!(test_res == test_eqs, false)
    }

    #[test]
    fn test_diseq_1() {
        let test_res = string_to_ts("(bar,baz)");
        let test_eqs = string_to_ts("bar,baz");
        assert_eq!(test_res == test_eqs, false)
    }

    #[test]
    fn test_is_empty() {
        let test0: TokenStream = Vec::<TokenTree>::new().into_iter().collect();
        let test1: TokenStream =
            TokenTree::Token(sp(0, 1), Token::Ident(Ident::from_str("a"))).into();
        let test2 = string_to_ts("foo(bar::baz)");

        assert_eq!(test0.is_empty(), true);
        assert_eq!(test1.is_empty(), false);
        assert_eq!(test2.is_empty(), false);
    }
}
