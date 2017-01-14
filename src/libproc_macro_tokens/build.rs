// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::Ident;
use syntax::codemap::DUMMY_SP;
use syntax::parse::token::{self, Token};
use syntax::symbol::keywords;
use syntax::tokenstream::{self, TokenTree, TokenStream};
use std::rc::Rc;

/// A wrapper around `TokenStream::concat` to avoid extra namespace specification and
/// provide TokenStream concatenation as a generic operator.
pub fn concat(ts1: TokenStream, ts2: TokenStream) -> TokenStream {
    TokenStream::concat(ts1, ts2)
}

/// Flatten a sequence of TokenStreams into a single TokenStream.
pub fn flatten<T: Iterator<Item=TokenStream>>(mut iter: T) -> TokenStream {
    match iter.next() {
        Some(mut ts) => {
            for next in iter {
                ts = TokenStream::concat(ts, next);
            }
            ts
        }
        None => TokenStream::mk_empty()
    }
}

/// Checks if two identifiers have the same name, disregarding context. This allows us to
/// fake 'reserved' keywords.
// FIXME We really want `free-identifier-=?` (a la Dybvig 1993). von Tander 2007 is
// probably the easiest way to do that.
pub fn ident_eq(tident: &TokenTree, id: Ident) -> bool {
    let tid = match *tident {
        TokenTree::Token(_, Token::Ident(ref id)) => id,
        _ => {
            return false;
        }
    };

    tid.name == id.name
}

// ____________________________________________________________________________________________
// Conversion operators

/// Convert a `&str` into a Token.
pub fn str_to_token_ident(s: &str) -> Token {
    Token::Ident(Ident::from_str(s))
}

/// Converts a keyword (from `syntax::parse::token::keywords`) into a Token that
/// corresponds to it.
pub fn keyword_to_token_ident(kw: keywords::Keyword) -> Token {
    Token::Ident(Ident::from_str(&kw.name().as_str()[..]))
}

// ____________________________________________________________________________________________
// Build Procedures

/// Generically takes a `ts` and delimiter and returns `ts` delimited by the specified
/// delimiter.
pub fn build_delimited(ts: TokenStream, delim: token::DelimToken) -> TokenStream {
    let tts = ts.to_tts();
    TokenStream::from_tts(vec![TokenTree::Delimited(DUMMY_SP,
                                                    Rc::new(tokenstream::Delimited {
                                                        delim: delim,
                                                        open_span: DUMMY_SP,
                                                        tts: tts,
                                                        close_span: DUMMY_SP,
                                                    }))])
}

/// Takes `ts` and returns `[ts]`.
pub fn build_bracket_delimited(ts: TokenStream) -> TokenStream {
    build_delimited(ts, token::DelimToken::Bracket)
}

/// Takes `ts` and returns `{ts}`.
pub fn build_brace_delimited(ts: TokenStream) -> TokenStream {
    build_delimited(ts, token::DelimToken::Brace)
}

/// Takes `ts` and returns `(ts)`.
pub fn build_paren_delimited(ts: TokenStream) -> TokenStream {
    build_delimited(ts, token::DelimToken::Paren)
}

/// Constructs `()`.
pub fn build_empty_args() -> TokenStream {
    build_paren_delimited(TokenStream::mk_empty())
}
