// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use codemap::{BytePos, spanned};
use parse::lexer::reader;
use parse::parser::Parser;
use parse::token;

use opt_vec;
use opt_vec::OptVec;

// SeqSep : a sequence separator (token)
// and whether a trailing separator is allowed.
pub struct SeqSep {
    sep: Option<token::Token>,
    trailing_sep_allowed: bool
}

pub fn seq_sep_trailing_disallowed(t: token::Token) -> SeqSep {
    SeqSep {
        sep: Some(t),
        trailing_sep_allowed: false,
    }
}
pub fn seq_sep_trailing_allowed(t: token::Token) -> SeqSep {
    SeqSep {
        sep: Some(t),
        trailing_sep_allowed: true,
    }
}
pub fn seq_sep_none() -> SeqSep {
    SeqSep {
        sep: None,
        trailing_sep_allowed: false,
    }
}

// maps any token back to a string. not necessary if you know it's
// an identifier....
pub fn token_to_str(reader: @reader, token: &token::Token) -> ~str {
    token::to_str(reader.interner(), token)
}

pub impl Parser {
    // convert a token to a string using self's reader
    fn token_to_str(&self, token: &token::Token) -> ~str {
        token::to_str(self.reader.interner(), token)
    }

    // convert the current token to a string using self's reader
    fn this_token_to_str(&self) -> ~str {
        self.token_to_str(self.token)
    }

    fn unexpected_last(&self, t: &token::Token) -> ! {
        self.span_fatal(
            *self.last_span,
            fmt!(
                "unexpected token: `%s`",
                self.token_to_str(t)
            )
        );
    }

    fn unexpected(&self) -> ! {
        self.fatal(
            fmt!(
                "unexpected token: `%s`",
                self.this_token_to_str()
            )
        );
    }

    // expect and consume the token t. Signal an error if
    // the next token is not t.
    fn expect(&self, t: &token::Token) {
        if *self.token == *t {
            self.bump();
        } else {
            self.fatal(
                fmt!(
                    "expected `%s` but found `%s`",
                    self.token_to_str(t),
                    self.this_token_to_str()
                )
            )
        }
    }

    fn parse_ident(&self) -> ast::ident {
        self.check_strict_keywords();
        self.check_reserved_keywords();
        match *self.token {
            token::IDENT(i, _) => {
                self.bump();
                i
            }
            token::INTERPOLATED(token::nt_ident(*)) => {
                self.bug(
                    ~"ident interpolation not converted to real token"
                );
            }
            _ => {
                self.fatal(
                    fmt!(
                        "expected ident, found `%s`",
                        self.this_token_to_str()
                    )
                );
            }
        }
    }

    fn parse_path_list_ident(&self) -> ast::path_list_ident {
        let lo = self.span.lo;
        let ident = self.parse_ident();
        let hi = self.last_span.hi;
        spanned(lo, hi, ast::path_list_ident_ { name: ident,
                                                id: self.get_id() })
    }

    // consume token 'tok' if it exists. Returns true if the given
    // token was present, false otherwise.
    fn eat(&self, tok: &token::Token) -> bool {
        return if *self.token == *tok { self.bump(); true } else { false };
    }

    // Storing keywords as interned idents instead of strings would be nifty.

    // A sanity check that the word we are asking for is a known keyword
    // NOTE: this could be done statically....
    fn require_keyword(&self, word: &str) {
        if !self.keywords.contains_equiv(&word) {
            self.bug(fmt!("unknown keyword: %s", word));
        }
    }

    // return true when this token represents the given string, and is not
    // followed immediately by :: .
    fn token_is_word(&self, word: &str, tok: &token::Token) -> bool {
        match *tok {
            token::IDENT(sid, false) => { word == *self.id_to_str(sid) }
             _ => { false }
        }
    }

    fn token_is_keyword(&self, word: &str, tok: &token::Token) -> bool {
        self.require_keyword(word);
        self.token_is_word(word, tok)
    }

    fn is_keyword(&self, word: &str) -> bool {
        self.token_is_keyword(word, &copy *self.token)
    }

    fn id_is_any_keyword(&self, id: ast::ident) -> bool {
        self.keywords.contains(self.id_to_str(id))
    }

    fn is_any_keyword(&self, tok: &token::Token) -> bool {
        match *tok {
          token::IDENT(sid, false) => {
            self.keywords.contains(self.id_to_str(sid))
          }
          _ => false
        }
    }

    // if the given word is not a keyword, signal an error.
    // if the next token is the given keyword, eat it and return
    // true. Otherwise, return false.
    fn eat_keyword(&self, word: &str) -> bool {
        self.require_keyword(word);
        let is_kw = match *self.token {
            token::IDENT(sid, false) => word == *self.id_to_str(sid),
            _ => false
        };
        if is_kw { self.bump() }
        is_kw
    }

    // if the given word is not a keyword, signal an error.
    // if the next token is not the given word, signal an error.
    // otherwise, eat it.
    fn expect_keyword(&self, word: &str) {
        self.require_keyword(word);
        if !self.eat_keyword(word) {
            self.fatal(
                fmt!(
                    "expected `%s`, found `%s`",
                    word,
                    self.this_token_to_str()
                )
            );
        }
    }

    // return true if the given string is a strict keyword
    fn is_strict_keyword(&self, word: &str) -> bool {
        self.strict_keywords.contains_equiv(&word)
    }

    // signal an error if the current token is a strict keyword
    fn check_strict_keywords(&self) {
        match *self.token {
            token::IDENT(_, false) => {
                let w = token_to_str(self.reader, &copy *self.token);
                self.check_strict_keywords_(w);
            }
            _ => ()
        }
    }

    // signal an error if the given string is a strict keyword
    fn check_strict_keywords_(&self, w: &str) {
        if self.is_strict_keyword(w) {
            self.span_err(*self.last_span,
                          fmt!("found `%s` in ident position", w));
        }
    }

    // return true if this is a reserved keyword
    fn is_reserved_keyword(&self, word: &str) -> bool {
        self.reserved_keywords.contains_equiv(&word)
    }

    // signal an error if the current token is a reserved keyword
    fn check_reserved_keywords(&self) {
        match *self.token {
            token::IDENT(_, false) => {
                let w = token_to_str(self.reader, &copy *self.token);
                self.check_reserved_keywords_(w);
            }
            _ => ()
        }
    }

    // signal an error if the given string is a reserved keyword
    fn check_reserved_keywords_(&self, w: &str) {
        if self.is_reserved_keyword(w) {
            self.fatal(fmt!("`%s` is a reserved keyword", w));
        }
    }

    // expect and consume a GT. if a >> is seen, replace it
    // with a single > and continue. If a GT is not seen,
    // signal an error.
    fn expect_gt(&self) {
        if *self.token == token::GT {
            self.bump();
        } else if *self.token == token::BINOP(token::SHR) {
            self.replace_token(
                token::GT,
                self.span.lo + BytePos(1u),
                self.span.hi
            );
        } else {
            let mut s: ~str = ~"expected `";
            s += self.token_to_str(&token::GT);
            s += ~"`, found `";
            s += self.this_token_to_str();
            s += ~"`";
            self.fatal(s);
        }
    }

    // parse a sequence bracketed by '<' and '>', stopping
    // before the '>'.
    fn parse_seq_to_before_gt<T: Copy>(
        &self,
        sep: Option<token::Token>,
        f: &fn(&Parser) -> T
    ) -> OptVec<T> {
        let mut first = true;
        let mut v = opt_vec::Empty;
        while *self.token != token::GT
            && *self.token != token::BINOP(token::SHR) {
            match sep {
              Some(ref t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            v.push(f(self));
        }
        return v;
    }

    fn parse_seq_to_gt<T: Copy>(
        &self,
        sep: Option<token::Token>,
        f: &fn(&Parser) -> T
    ) -> OptVec<T> {
        let v = self.parse_seq_to_before_gt(sep, f);
        self.expect_gt();
        return v;
    }

    // parse a sequence, including the closing delimiter. The function
    // f must consume tokens until reaching the next separator or
    // closing bracket.
    fn parse_seq_to_end<T: Copy>(
        &self,
        ket: &token::Token,
        sep: SeqSep,
        f: &fn(&Parser) -> T
    ) -> ~[T] {
        let val = self.parse_seq_to_before_end(ket, sep, f);
        self.bump();
        val
    }

    // parse a sequence, not including the closing delimiter. The function
    // f must consume tokens until reaching the next separator or
    // closing bracket.
    fn parse_seq_to_before_end<T: Copy>(
        &self,
        ket: &token::Token,
        sep: SeqSep,
        f: &fn(&Parser) -> T
    ) -> ~[T] {
        let mut first: bool = true;
        let mut v: ~[T] = ~[];
        while *self.token != *ket {
            match sep.sep {
              Some(ref t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            if sep.trailing_sep_allowed && *self.token == *ket { break; }
            v.push(f(self));
        }
        return v;
    }

    // parse a sequence, including the closing delimiter. The function
    // f must consume tokens until reaching the next separator or
    // closing bracket.
    fn parse_unspanned_seq<T: Copy>(
        &self,
        bra: &token::Token,
        ket: &token::Token,
        sep: SeqSep,
        f: &fn(&Parser) -> T
    ) -> ~[T] {
        self.expect(bra);
        let result = self.parse_seq_to_before_end(ket, sep, f);
        self.bump();
        result
    }

    // NB: Do not use this function unless you actually plan to place the
    // spanned list in the AST.
    fn parse_seq<T: Copy>(
        &self,
        bra: &token::Token,
        ket: &token::Token,
        sep: SeqSep,
        f: &fn(&Parser) -> T
    ) -> spanned<~[T]> {
        let lo = self.span.lo;
        self.expect(bra);
        let result = self.parse_seq_to_before_end(ket, sep, f);
        let hi = self.span.hi;
        self.bump();
        spanned(lo, hi, result)
    }
}
