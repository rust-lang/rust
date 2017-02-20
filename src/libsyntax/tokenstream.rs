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
//! TokenStreams represent syntactic objects before they are converted into ASTs.
//! A `TokenStream` is, roughly speaking, a sequence (eg stream) of `TokenTree`s,
//! which are themselves a single `Token` or a `Delimited` subsequence of tokens.
//!
//! ## Ownership
//! TokenStreams are persistent data structures constructed as ropes with reference
//! counted-children. In general, this means that calling an operation on a TokenStream
//! (such as `slice`) produces an entirely new TokenStream from the borrowed reference to
//! the original. This essentially coerces TokenStreams into 'views' of their subparts,
//! and a borrowed TokenStream is sufficient to build an owned TokenStream without taking
//! ownership of the original.

use ast::{self, LitKind};
use syntax_pos::{BytePos, Span, DUMMY_SP};
use codemap::Spanned;
use ext::base;
use ext::tt::{macro_parser, quoted};
use parse::{self, Directory};
use parse::token::{self, Token, Lit};
use print::pprust;
use serialize::{Decoder, Decodable, Encoder, Encodable};
use symbol::Symbol;
use util::RcSlice;

use std::{fmt, iter, mem};
use std::rc::Rc;

/// A delimited sequence of token trees
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub struct Delimited {
    /// The type of delimiter
    pub delim: token::DelimToken,
    /// The delimited sequence of token trees
    pub tts: Vec<TokenTree>,
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
        let open_span = match span {
            DUMMY_SP => DUMMY_SP,
            _ => Span { hi: span.lo + BytePos(self.delim.len() as u32), ..span },
        };
        TokenTree::Token(open_span, self.open_token())
    }

    /// Returns the closing delimiter as a token tree.
    pub fn close_tt(&self, span: Span) -> TokenTree {
        let close_span = match span {
            DUMMY_SP => DUMMY_SP,
            _ => Span { lo: span.hi - BytePos(self.delim.len() as u32), ..span },
        };
        TokenTree::Token(close_span, self.close_token())
    }

    /// Returns the token trees inside the delimiters.
    pub fn subtrees(&self) -> &[TokenTree] {
        &self.tts
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
/// macro_parser::matched_nonterminals into the `SubstNt`s it finds.
///
/// The RHS of an MBE macro is the only place `SubstNt`s are substituted.
/// Nothing special happens to misnamed or misplaced `SubstNt`s.
#[derive(Debug, Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash)]
pub enum TokenTree {
    /// A single token
    Token(Span, token::Token),
    /// A delimited sequence of token trees
    Delimited(Span, Rc<Delimited>),
}

impl TokenTree {
    /// Use this token tree as a matcher to parse given tts.
    pub fn parse(cx: &base::ExtCtxt, mtch: &[quoted::TokenTree], tts: &[TokenTree])
                 -> macro_parser::NamedParseResult {
        // `None` is because we're not interpolating
        let directory = Directory {
            path: cx.current_expansion.module.directory.clone(),
            ownership: cx.current_expansion.directory_ownership,
        };
        macro_parser::parse(cx.parse_sess(), tts.iter().cloned().collect(), mtch, Some(directory))
    }

    /// Check if this TokenTree is equal to the other, regardless of span information.
    pub fn eq_unspanned(&self, other: &TokenTree) -> bool {
        match (self, other) {
            (&TokenTree::Token(_, ref tk), &TokenTree::Token(_, ref tk2)) => tk == tk2,
            (&TokenTree::Delimited(_, ref dl), &TokenTree::Delimited(_, ref dl2)) => {
                (*dl).delim == (*dl2).delim && dl.tts.len() == dl2.tts.len() &&
                {
                    for (tt1, tt2) in dl.tts.iter().zip(dl2.tts.iter()) {
                        if !tt1.eq_unspanned(tt2) {
                            return false;
                        }
                    }
                    true
                }
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

    /// Indicates if the token is an identifier.
    pub fn is_ident(&self) -> bool {
        self.maybe_ident().is_some()
    }

    /// Returns an identifier.
    pub fn maybe_ident(&self) -> Option<ast::Ident> {
        match *self {
            TokenTree::Token(_, Token::Ident(t)) => Some(t.clone()),
            TokenTree::Delimited(_, ref dl) => {
                let tts = dl.subtrees();
                if tts.len() != 1 {
                    return None;
                }
                tts[0].maybe_ident()
            }
            _ => None,
        }
    }

    /// Returns a Token literal.
    pub fn maybe_lit(&self) -> Option<token::Lit> {
        match *self {
            TokenTree::Token(_, Token::Literal(l, _)) => Some(l.clone()),
            TokenTree::Delimited(_, ref dl) => {
                let tts = dl.subtrees();
                if tts.len() != 1 {
                    return None;
                }
                tts[0].maybe_lit()
            }
            _ => None,
        }
    }

    /// Returns an AST string literal.
    pub fn maybe_str(&self) -> Option<ast::Lit> {
        match *self {
            TokenTree::Token(sp, Token::Literal(Lit::Str_(s), _)) => {
                let l = LitKind::Str(Symbol::intern(&parse::str_lit(&s.as_str())),
                                     ast::StrStyle::Cooked);
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            TokenTree::Token(sp, Token::Literal(Lit::StrRaw(s, n), _)) => {
                let l = LitKind::Str(Symbol::intern(&parse::raw_str_lit(&s.as_str())),
                                     ast::StrStyle::Raw(n));
                Some(Spanned {
                    node: l,
                    span: sp,
                })
            }
            _ => None,
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

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&pprust::tts_to_string(&self.trees().collect::<Vec<_>>()))
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


#[cfg(test)]
mod tests {
    use super::*;
    use syntax::ast::Ident;
    use syntax_pos::{Span, BytePos, NO_EXPANSION};
    use parse::token::Token;
    use util::parser_testing::string_to_tts;

    fn string_to_ts(string: &str) -> TokenStream {
        string_to_tts(string.to_owned()).into_iter().collect()
    }

    fn sp(a: u32, b: u32) -> Span {
        Span {
            lo: BytePos(a),
            hi: BytePos(b),
            expn_id: NO_EXPANSION,
        }
    }

    #[test]
    fn test_concat() {
        let test_res = string_to_ts("foo::bar::baz");
        let test_fst = string_to_ts("foo::bar");
        let test_snd = string_to_ts("::baz");
        let eq_res = TokenStream::concat([test_fst, test_snd].iter().cloned());
        assert_eq!(test_res.trees().count(), 5);
        assert_eq!(eq_res.trees().count(), 5);
        assert_eq!(test_res.eq_unspanned(&eq_res), true);
    }

    #[test]
    fn test_from_to_bijection() {
        let test_start = string_to_tts("foo::bar(baz)".to_string());
        let ts = test_start.iter().cloned().collect::<TokenStream>();
        let test_end: Vec<TokenTree> = ts.trees().collect();
        assert_eq!(test_start, test_end)
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
