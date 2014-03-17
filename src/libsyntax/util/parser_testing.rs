// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast;
use parse::{new_parse_sess};
use parse::{ParseSess,string_to_filemap,filemap_to_tts};
use parse::{new_parser_from_source_str};
use parse::parser::Parser;
use parse::token;

use std::vec_ng::Vec;

// map a string to tts, using a made-up filename:
pub fn string_to_tts(source_str: ~str) -> Vec<ast::TokenTree> {
    let ps = new_parse_sess();
    filemap_to_tts(&ps, string_to_filemap(&ps, source_str,~"bogofile"))
}

// map string to parser (via tts)
pub fn string_to_parser<'a>(ps: &'a ParseSess, source_str: ~str) -> Parser<'a> {
    new_parser_from_source_str(ps, Vec::new(), ~"bogofile", source_str)
}

fn with_error_checking_parse<T>(s: ~str, f: |&mut Parser| -> T) -> T {
    let ps = new_parse_sess();
    let mut p = string_to_parser(&ps, s);
    let x = f(&mut p);
    p.abort_if_errors();
    x
}

// parse a string, return a crate.
pub fn string_to_crate (source_str : ~str) -> ast::Crate {
    with_error_checking_parse(source_str, |p| {
        p.parse_crate_mod()
    })
}

// parse a string, return an expr
pub fn string_to_expr (source_str : ~str) -> @ast::Expr {
    with_error_checking_parse(source_str, |p| {
        p.parse_expr()
    })
}

// parse a string, return an item
pub fn string_to_item (source_str : ~str) -> Option<@ast::Item> {
    with_error_checking_parse(source_str, |p| {
        p.parse_item(Vec::new())
    })
}

// parse a string, return a stmt
pub fn string_to_stmt(source_str : ~str) -> @ast::Stmt {
    with_error_checking_parse(source_str, |p| {
        p.parse_stmt(Vec::new())
    })
}

// parse a string, return a pat. Uses "irrefutable"... which doesn't
// (currently) affect parsing.
pub fn string_to_pat(source_str: ~str) -> @ast::Pat {
    string_to_parser(&new_parse_sess(), source_str).parse_pat()
}

// convert a vector of strings to a vector of ast::Ident's
pub fn strs_to_idents(ids: Vec<&str> ) -> Vec<ast::Ident> {
    ids.map(|u| token::str_to_ident(*u))
}

// does the given string match the pattern? whitespace in the first string
// may be deleted or replaced with other whitespace to match the pattern.
// this function is unicode-ignorant; fortunately, the careful design of
// UTF-8 mitigates this ignorance.  In particular, this function only collapses
// sequences of \n, \r, ' ', and \t, but it should otherwise tolerate unicode
// chars. Unsurprisingly, it doesn't do NKF-normalization(?).
pub fn matches_codepattern(a : &str, b : &str) -> bool {
    let mut idx_a = 0;
    let mut idx_b = 0;
    loop {
        if idx_a == a.len() && idx_b == b.len() {
            return true;
        }
        else if idx_a == a.len() {return false;}
        else if idx_b == b.len() {
            // maybe the stuff left in a is all ws?
            if is_whitespace(a.char_at(idx_a)) {
                return scan_for_non_ws_or_end(a,idx_a) == a.len();
            } else {
                return false;
            }
        }
        // ws in both given and pattern:
        else if is_whitespace(a.char_at(idx_a))
           && is_whitespace(b.char_at(idx_b)) {
            idx_a = scan_for_non_ws_or_end(a,idx_a);
            idx_b = scan_for_non_ws_or_end(b,idx_b);
        }
        // ws in given only:
        else if is_whitespace(a.char_at(idx_a)) {
            idx_a = scan_for_non_ws_or_end(a,idx_a);
        }
        // *don't* silently eat ws in expected only.
        else if a.char_at(idx_a) == b.char_at(idx_b) {
            idx_a += 1;
            idx_b += 1;
        }
        else {
            return false;
        }
    }
}

// given a string and an index, return the first uint >= idx
// that is a non-ws-char or is outside of the legal range of
// the string.
fn scan_for_non_ws_or_end(a : &str, idx: uint) -> uint {
    let mut i = idx;
    let len = a.len();
    while (i < len) && (is_whitespace(a.char_at(i))) {
        i += 1;
    }
    i
}

// copied from lexer.
pub fn is_whitespace(c: char) -> bool {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

#[cfg(test)]
mod test {
    use super::*;

    #[test] fn eqmodws() {
        assert_eq!(matches_codepattern("",""),true);
        assert_eq!(matches_codepattern("","a"),false);
        assert_eq!(matches_codepattern("a",""),false);
        assert_eq!(matches_codepattern("a","a"),true);
        assert_eq!(matches_codepattern("a b","a   \n\t\r  b"),true);
        assert_eq!(matches_codepattern("a b ","a   \n\t\r  b"),true);
        assert_eq!(matches_codepattern("a b","a   \n\t\r  b "),false);
        assert_eq!(matches_codepattern("a   b","a b"),true);
        assert_eq!(matches_codepattern("ab","a b"),false);
        assert_eq!(matches_codepattern("a   b","ab"),true);
    }
}
