import std::map::{hashmap};
import ast_util::spanned;
import parser::parser;
import lexer::reader;

type seq_sep = {
    sep: option<token::token>,
    trailing_sep_allowed: bool
};

fn seq_sep_trailing_disallowed(t: token::token) -> seq_sep {
    return {sep: option::some(t), trailing_sep_allowed: false};
}
fn seq_sep_trailing_allowed(t: token::token) -> seq_sep {
    return {sep: option::some(t), trailing_sep_allowed: true};
}
fn seq_sep_none() -> seq_sep {
    return {sep: option::none, trailing_sep_allowed: false};
}

fn token_to_str(reader: reader, ++token: token::token) -> ~str {
    token::to_str(*reader.interner(), token)
}

trait parser_common {
    fn unexpected_last(t: token::token) -> !;
    fn unexpected() -> !;
    fn expect(t: token::token);
    fn parse_ident() -> ast::ident;
    fn parse_path_list_ident() -> ast::path_list_ident;
    fn parse_value_ident() -> ast::ident;
    fn eat(tok: token::token) -> bool;
    // A sanity check that the word we are asking for is a known keyword
    fn require_keyword(word: ~str);
    fn token_is_keyword(word: ~str, ++tok: token::token) -> bool;
    fn is_keyword(word: ~str) -> bool;
    fn is_any_keyword(tok: token::token) -> bool;
    fn eat_keyword(word: ~str) -> bool;
    fn expect_keyword(word: ~str);
    fn is_restricted_keyword(word: ~str) -> bool;
    fn check_restricted_keywords();
    fn check_restricted_keywords_(w: ~str);
    fn expect_gt();
    fn parse_seq_to_before_gt<T: copy>(sep: option<token::token>,
                                       f: fn(parser) -> T) -> ~[T];
    fn parse_seq_to_gt<T: copy>(sep: option<token::token>,
                                f: fn(parser) -> T) -> ~[T];
    fn parse_seq_lt_gt<T: copy>(sep: option<token::token>,
                                f: fn(parser) -> T) -> spanned<~[T]>;
    fn parse_seq_to_end<T: copy>(ket: token::token, sep: seq_sep,
                                 f: fn(parser) -> T) -> ~[T];
    fn parse_seq_to_before_end<T: copy>(ket: token::token, sep: seq_sep,
                                        f: fn(parser) -> T) -> ~[T];
    fn parse_unspanned_seq<T: copy>(bra: token::token,
                                    ket: token::token,
                                    sep: seq_sep,
                                    f: fn(parser) -> T) -> ~[T];
    fn parse_seq<T: copy>(bra: token::token, ket: token::token, sep: seq_sep,
                          f: fn(parser) -> T) -> spanned<~[T]>;
}

impl parser_common of parser_common for parser {
    fn unexpected_last(t: token::token) -> ! {
        self.span_fatal(
            copy self.last_span,
            ~"unexpected token: `" + token_to_str(self.reader, t) + ~"`");
    }

    fn unexpected() -> ! {
        self.fatal(~"unexpected token: `"
                   + token_to_str(self.reader, self.token) + ~"`");
    }

    fn expect(t: token::token) {
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
        alt copy self.token {
          token::IDENT(i, _) => { self.bump(); return self.get_str(i); }
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
        self.check_restricted_keywords();
        return self.parse_ident();
    }

    fn eat(tok: token::token) -> bool {
        return if self.token == tok { self.bump(); true } else { false };
    }

    // A sanity check that the word we are asking for is a known keyword
    fn require_keyword(word: ~str) {
        if !self.keywords.contains_key_ref(&word) {
            self.bug(fmt!{"unknown keyword: %s", word});
        }
    }

    fn token_is_word(word: ~str, ++tok: token::token) -> bool {
        alt tok {
          token::IDENT(sid, false) => { word == *self.get_str(sid) }
          _ => { false }
        }
    }

    fn token_is_keyword(word: ~str, ++tok: token::token) -> bool {
        self.require_keyword(word);
        self.token_is_word(word, tok)
    }

    fn is_keyword(word: ~str) -> bool {
        self.token_is_keyword(word, self.token)
    }

    fn is_any_keyword(tok: token::token) -> bool {
        alt tok {
          token::IDENT(sid, false) => {
            self.keywords.contains_key_ref(self.get_str(sid))
          }
          _ => false
        }
    }

    fn eat_keyword(word: ~str) -> bool {
        self.require_keyword(word);

        let mut bump = false;
        let val = alt self.token {
          token::IDENT(sid, false) => {
            if word == *self.get_str(sid) {
                bump = true;
                true
            } else { false }
          }
          _ => false
        };
        if bump { self.bump() }
        val
    }

    fn expect_keyword(word: ~str) {
        self.require_keyword(word);
        if !self.eat_keyword(word) {
            self.fatal(~"expected `" + word + ~"`, found `" +
                       token_to_str(self.reader, self.token) +
                       ~"`");
        }
    }

    fn is_restricted_keyword(word: ~str) -> bool {
        self.restricted_keywords.contains_key_ref(&word)
    }

    fn check_restricted_keywords() {
        alt self.token {
          token::IDENT(_, false) => {
            let w = token_to_str(self.reader, self.token);
            self.check_restricted_keywords_(w);
          }
          _ => ()
        }
    }

    fn check_restricted_keywords_(w: ~str) {
        if self.is_restricted_keyword(w) {
            self.fatal(~"found `" + w + ~"` in restricted position");
        }
    }

    fn expect_gt() {
        if self.token == token::GT {
            self.bump();
        } else if self.token == token::BINOP(token::SHR) {
            self.swap(token::GT, self.span.lo + 1u, self.span.hi);
        } else {
            let mut s: ~str = ~"expected `";
            s += token_to_str(self.reader, token::GT);
            s += ~"`, found `";
            s += token_to_str(self.reader, self.token);
            s += ~"`";
            self.fatal(s);
        }
    }

    fn parse_seq_to_before_gt<T: copy>(sep: option<token::token>,
                                       f: fn(parser) -> T) -> ~[T] {
        let mut first = true;
        let mut v = ~[];
        while self.token != token::GT
            && self.token != token::BINOP(token::SHR) {
            alt sep {
              some(t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            vec::push(v, f(self));
        }

        return v;
    }

    fn parse_seq_to_gt<T: copy>(sep: option<token::token>,
                                f: fn(parser) -> T) -> ~[T] {
        let v = self.parse_seq_to_before_gt(sep, f);
        self.expect_gt();

        return v;
    }

    fn parse_seq_lt_gt<T: copy>(sep: option<token::token>,
                                f: fn(parser) -> T) -> spanned<~[T]> {
        let lo = self.span.lo;
        self.expect(token::LT);
        let result = self.parse_seq_to_before_gt::<T>(sep, f);
        let hi = self.span.hi;
        self.expect_gt();
        return spanned(lo, hi, result);
    }

    fn parse_seq_to_end<T: copy>(ket: token::token, sep: seq_sep,
                                 f: fn(parser) -> T) -> ~[T] {
        let val = self.parse_seq_to_before_end(ket, sep, f);
        self.bump();
        return val;
    }


    fn parse_seq_to_before_end<T: copy>(ket: token::token, sep: seq_sep,
                                        f: fn(parser) -> T) -> ~[T] {
        let mut first: bool = true;
        let mut v: ~[T] = ~[];
        while self.token != ket {
            alt sep.sep {
              some(t) => {
                if first { first = false; }
                else { self.expect(t); }
              }
              _ => ()
            }
            if sep.trailing_sep_allowed && self.token == ket { break; }
            vec::push(v, f(self));
        }
        return v;
    }

    fn parse_unspanned_seq<T: copy>(bra: token::token,
                                    ket: token::token,
                                    sep: seq_sep,
                                    f: fn(parser) -> T) -> ~[T] {
        self.expect(bra);
        let result = self.parse_seq_to_before_end::<T>(ket, sep, f);
        self.bump();
        return result;
    }

    // NB: Do not use this function unless you actually plan to place the
    // spanned list in the AST.
    fn parse_seq<T: copy>(bra: token::token, ket: token::token, sep: seq_sep,
                          f: fn(parser) -> T) -> spanned<~[T]> {
        let lo = self.span.lo;
        self.expect(bra);
        let result = self.parse_seq_to_before_end::<T>(ket, sep, f);
        let hi = self.span.hi;
        self.bump();
        return spanned(lo, hi, result);
    }
}
