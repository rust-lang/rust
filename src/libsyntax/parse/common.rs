use std::map::{HashMap};
use ast_util::spanned;
use parser::Parser;
use lexer::reader;

type seq_sep = {
    sep: Option<token::Token>,
    trailing_sep_allowed: bool
};

fn seq_sep_trailing_disallowed(t: token::Token) -> seq_sep {
    return {sep: option::Some(t), trailing_sep_allowed: false};
}
fn seq_sep_trailing_allowed(t: token::Token) -> seq_sep {
    return {sep: option::Some(t), trailing_sep_allowed: true};
}
fn seq_sep_none() -> seq_sep {
    return {sep: option::None, trailing_sep_allowed: false};
}

fn token_to_str(reader: reader, ++token: token::Token) -> ~str {
    token::to_str(reader.interner(), token)
}

impl Parser {
    fn unexpected_last(t: token::Token) -> ! {
        self.span_fatal(
            copy self.last_span,
            ~"unexpected token: `" + token_to_str(self.reader, t) + ~"`");
    }

    fn unexpected() -> ! {
        self.fatal(~"unexpected token: `"
                   + token_to_str(self.reader, self.token) + ~"`");
    }

    fn expect(t: token::Token) {
        if self.token == t {
            self.bump();
        } else {
            let mut s: ~str = ~"expected `";
            s += token_to_str(self.reader, t);
            s += ~"` but found `";
            s += token_to_str(self.reader, self.token);
            self.fatal(s + ~"`");
        }
    }

    fn parse_ident() -> ast::ident {
        self.check_strict_keywords();
        self.check_reserved_keywords();
        match copy self.token {
          token::IDENT(i, _) => { self.bump(); return i; }
          token::INTERPOLATED(token::nt_ident(*)) => { self.bug(
              ~"ident interpolation not converted to real token"); }
          _ => { self.fatal(~"expected ident, found `"
                         + token_to_str(self.reader, self.token)
                         + ~"`"); }
        }
    }

    fn parse_path_list_ident() -> ast::path_list_ident {
        let lo = self.span.lo;
        let ident = self.parse_ident();
        let hi = self.span.hi;
        return spanned(lo, hi, {name: ident, id: self.get_id()});
    }

    fn parse_value_ident() -> ast::ident {
        return self.parse_ident();
    }

    fn eat(tok: token::Token) -> bool {
        return if self.token == tok { self.bump(); true } else { false };
    }

    // Storing keywords as interned idents instead of strings would be nifty.

    // A sanity check that the word we are asking for is a known keyword
    fn require_keyword(word: ~str) {
        if !self.keywords.contains_key_ref(&word) {
            self.bug(fmt!("unknown keyword: %s", word));
        }
    }

    fn token_is_word(word: ~str, ++tok: token::Token) -> bool {
        match tok {
          token::IDENT(sid, false) => { *self.id_to_str(sid) == word }
          _ => { false }
        }
    }

    fn token_is_keyword(word: ~str, ++tok: token::Token) -> bool {
        self.require_keyword(word);
        self.token_is_word(word, tok)
    }

    fn is_keyword(word: ~str) -> bool {
        self.token_is_keyword(word, self.token)
    }

    fn is_any_keyword(tok: token::Token) -> bool {
        match tok {
          token::IDENT(sid, false) => {
            self.keywords.contains_key_ref(self.id_to_str(sid))
          }
          _ => false
        }
    }

    fn eat_keyword(word: ~str) -> bool {
        self.require_keyword(word);
        let is_kw = match self.token {
          token::IDENT(sid, false) => (word == *self.id_to_str(sid)),
          _ => false
        };
        if is_kw { self.bump() }
        is_kw
    }

    fn expect_keyword(word: ~str) {
        self.require_keyword(word);
        if !self.eat_keyword(word) {
            self.fatal(~"expected `" + word + ~"`, found `" +
                       token_to_str(self.reader, self.token) +
                       ~"`");
        }
    }

    fn is_strict_keyword(word: ~str) -> bool {
        self.strict_keywords.contains_key_ref(&word)
    }

    fn check_strict_keywords() {
        match self.token {
          token::IDENT(_, false) => {
            let w = token_to_str(self.reader, self.token);
            self.check_strict_keywords_(w);
          }
          _ => ()
        }
    }

    fn check_strict_keywords_(w: ~str) {
        if self.is_strict_keyword(w) {
            self.fatal(~"found `" + w + ~"` in ident position");
        }
    }

    fn is_reserved_keyword(word: ~str) -> bool {
        self.reserved_keywords.contains_key_ref(&word)
    }

    fn check_reserved_keywords() {
        match self.token {
          token::IDENT(_, false) => {
            let w = token_to_str(self.reader, self.token);
            self.check_reserved_keywords_(w);
          }
          _ => ()
        }
    }

    fn check_reserved_keywords_(w: ~str) {
        if self.is_reserved_keyword(w) {
            self.fatal(~"`" + w + ~"` is a reserved keyword");
        }
    }

    fn expect_gt() {
        if self.token == token::GT {
            self.bump();
        } else if self.token == token::BINOP(token::SHR) {
            self.swap(token::GT, self.span.lo + BytePos(1u), self.span.hi);
        } else {
            let mut s: ~str = ~"expected `";
            s += token_to_str(self.reader, token::GT);
            s += ~"`, found `";
            s += token_to_str(self.reader, self.token);
            s += ~"`";
            self.fatal(s);
        }
    }

    fn parse_seq_to_before_gt<T: Copy>(sep: Option<token::Token>,
                                       f: fn(Parser) -> T) -> ~[T] {
        let mut first = true;
        let mut v = ~[];
        while self.token != token::GT
            && self.token != token::BINOP(token::SHR) {
            match sep {
              Some(t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            v.push(f(self));
        }

        return v;
    }

    fn parse_seq_to_gt<T: Copy>(sep: Option<token::Token>,
                                f: fn(Parser) -> T) -> ~[T] {
        let v = self.parse_seq_to_before_gt(sep, f);
        self.expect_gt();

        return v;
    }

    fn parse_seq_lt_gt<T: Copy>(sep: Option<token::Token>,
                                f: fn(Parser) -> T) -> spanned<~[T]> {
        let lo = self.span.lo;
        self.expect(token::LT);
        let result = self.parse_seq_to_before_gt::<T>(sep, f);
        let hi = self.span.hi;
        self.expect_gt();
        return spanned(lo, hi, result);
    }

    fn parse_seq_to_end<T: Copy>(ket: token::Token, sep: seq_sep,
                                 f: fn(Parser) -> T) -> ~[T] {
        let val = self.parse_seq_to_before_end(ket, sep, f);
        self.bump();
        return val;
    }


    fn parse_seq_to_before_end<T: Copy>(ket: token::Token, sep: seq_sep,
                                        f: fn(Parser) -> T) -> ~[T] {
        let mut first: bool = true;
        let mut v: ~[T] = ~[];
        while self.token != ket {
            match sep.sep {
              Some(t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            if sep.trailing_sep_allowed && self.token == ket { break; }
            v.push(f(self));
        }
        return v;
    }

    fn parse_unspanned_seq<T: Copy>(bra: token::Token,
                                    ket: token::Token,
                                    sep: seq_sep,
                                    f: fn(Parser) -> T) -> ~[T] {
        self.expect(bra);
        let result = self.parse_seq_to_before_end::<T>(ket, sep, f);
        self.bump();
        return result;
    }

    // NB: Do not use this function unless you actually plan to place the
    // spanned list in the AST.
    fn parse_seq<T: Copy>(bra: token::Token, ket: token::Token, sep: seq_sep,
                          f: fn(Parser) -> T) -> spanned<~[T]> {
        let lo = self.span.lo;
        self.expect(bra);
        let result = self.parse_seq_to_before_end::<T>(ket, sep, f);
        let hi = self.span.hi;
        self.bump();
        return spanned(lo, hi, result);
    }
}
