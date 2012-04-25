import std::map::{hashmap};
import ast_util::spanned;
import parser::parser;

fn token_to_str(reader: reader, token: token::token) -> str {
    token::to_str(*reader.interner, token)
}

fn unexpected_last(p: parser, t: token::token) -> ! {
    p.span_fatal(p.last_span,
                 "unexpected token: '" + token_to_str(p.reader, t) + "'");
}

fn unexpected(p: parser) -> ! {
    p.fatal("unexpected token: '" + token_to_str(p.reader, p.token) + "'");
}

fn expect(p: parser, t: token::token) {
    if p.token == t {
        p.bump();
    } else {
        let mut s: str = "expecting '";
        s += token_to_str(p.reader, t);
        s += "' but found '";
        s += token_to_str(p.reader, p.token);
        p.fatal(s + "'");
    }
}

fn parse_ident(p: parser) -> ast::ident {
    alt p.token {
      token::IDENT(i, _) { p.bump(); ret p.get_str(i); }
      _ { p.fatal("expecting ident, found "
                  + token_to_str(p.reader, p.token)); }
    }
}

fn parse_path_list_ident(p: parser) -> ast::path_list_ident {
    let lo = p.span.lo;
    let ident = parse_ident(p);
    let hi = p.span.hi;
    ret spanned(lo, hi, {name: ident, id: p.get_id()});
}

fn parse_value_ident(p: parser) -> ast::ident {
    check_bad_expr_word(p);
    ret parse_ident(p);
}

fn eat(p: parser, tok: token::token) -> bool {
    ret if p.token == tok { p.bump(); true } else { false };
}

// A sanity check that the word we are asking for is a known keyword
fn require_keyword(p: parser, word: str) {
    if !p.keywords.contains_key(word) {
        p.bug(#fmt("unknown keyword: %s", word));
    }
}

fn is_keyword(p: parser, word: str) -> bool {
    require_keyword(p, word);
    ret alt p.token {
          token::IDENT(sid, false) { str::eq(word, p.get_str(sid)) }
          _ { false }
        };
}

fn eat_keyword(p: parser, word: str) -> bool {
    require_keyword(p, word);
    alt p.token {
      token::IDENT(sid, false) {
        if str::eq(word, p.get_str(sid)) {
            p.bump();
            ret true;
        } else { ret false; }
      }
      _ { ret false; }
    }
}

fn expect_keyword(p: parser, word: str) {
    require_keyword(p, word);
    if !eat_keyword(p, word) {
        p.fatal("expecting " + word + ", found " +
                    token_to_str(p.reader, p.token));
    }
}

fn is_bad_expr_word(p: parser, word: str) -> bool {
    p.bad_expr_words.contains_key(word)
}

fn check_bad_expr_word(p: parser) {
    alt p.token {
      token::IDENT(_, false) {
        let w = token_to_str(p.reader, p.token);
        if is_bad_expr_word(p, w) {
            p.fatal("found `" + w + "` in expression position");
        }
      }
      _ { }
    }
}

fn expect_gt(p: parser) {
    if p.token == token::GT {
        p.bump();
    } else if p.token == token::BINOP(token::LSR) {
        p.swap(token::GT, p.span.lo + 1u, p.span.hi);
    } else if p.token == token::BINOP(token::ASR) {
        p.swap(token::BINOP(token::LSR), p.span.lo + 1u, p.span.hi);
    } else {
        let mut s: str = "expecting ";
        s += token_to_str(p.reader, token::GT);
        s += ", found ";
        s += token_to_str(p.reader, p.token);
        p.fatal(s);
    }
}

fn parse_seq_to_before_gt<T: copy>(sep: option<token::token>,
                                  f: fn(parser) -> T,
                                  p: parser) -> [T] {
    let mut first = true;
    let mut v = [];
    while p.token != token::GT && p.token != token::BINOP(token::LSR) &&
              p.token != token::BINOP(token::ASR) {
        alt sep {
          some(t) { if first { first = false; } else { expect(p, t); } }
          _ { }
        }
        v += [f(p)];
    }

    ret v;
}

fn parse_seq_to_gt<T: copy>(sep: option<token::token>,
                           f: fn(parser) -> T, p: parser) -> [T] {
    let v = parse_seq_to_before_gt(sep, f, p);
    expect_gt(p);

    ret v;
}

fn parse_seq_lt_gt<T: copy>(sep: option<token::token>,
                           f: fn(parser) -> T,
                           p: parser) -> spanned<[T]> {
    let lo = p.span.lo;
    expect(p, token::LT);
    let result = parse_seq_to_before_gt::<T>(sep, f, p);
    let hi = p.span.hi;
    expect_gt(p);
    ret spanned(lo, hi, result);
}

fn parse_seq_to_end<T: copy>(ket: token::token, sep: seq_sep,
                            f: fn(parser) -> T, p: parser) -> [T] {
    let val = parse_seq_to_before_end(ket, sep, f, p);
    p.bump();
    ret val;
}

type seq_sep = {
    sep: option<token::token>,
    trailing_opt: bool   // is trailing separator optional?
};

fn seq_sep(t: token::token) -> seq_sep {
    ret {sep: option::some(t), trailing_opt: false};
}
fn seq_sep_opt(t: token::token) -> seq_sep {
    ret {sep: option::some(t), trailing_opt: true};
}
fn seq_sep_none() -> seq_sep {
    ret {sep: option::none, trailing_opt: false};
}

fn parse_seq_to_before_end<T: copy>(ket: token::token,
                                   sep: seq_sep,
                                   f: fn(parser) -> T, p: parser) -> [T] {
    let mut first: bool = true;
    let mut v: [T] = [];
    while p.token != ket {
        alt sep.sep {
          some(t) { if first { first = false; } else { expect(p, t); } }
          _ { }
        }
        if sep.trailing_opt && p.token == ket { break; }
        v += [f(p)];
    }
    ret v;
}

fn parse_seq<T: copy>(bra: token::token, ket: token::token,
                     sep: seq_sep, f: fn(parser) -> T,
                     p: parser) -> spanned<[T]> {
    let lo = p.span.lo;
    expect(p, bra);
    let result = parse_seq_to_before_end::<T>(ket, sep, f, p);
    let hi = p.span.hi;
    p.bump();
    ret spanned(lo, hi, result);
}
