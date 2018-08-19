// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{self, Ident};
use source_map::FilePathMapping;
use parse::{ParseSess, PResult, source_file_to_stream};
use parse::{lexer, new_parser_from_source_str};
use parse::parser::Parser;
use ptr::P;
use tokenstream::TokenStream;
use std::iter::Peekable;
use std::path::PathBuf;

/// Map a string to tts, using a made-up filename:
pub fn string_to_stream(source_str: String) -> TokenStream {
    let ps = ParseSess::new(FilePathMapping::empty());
    source_file_to_stream(&ps, ps.source_map()
                             .new_source_file(PathBuf::from("bogofile").into(), source_str), None)
}

/// Map string to parser (via tts)
pub fn string_to_parser<'a>(ps: &'a ParseSess, source_str: String) -> Parser<'a> {
    new_parser_from_source_str(ps, PathBuf::from("bogofile").into(), source_str)
}

fn with_error_checking_parse<'a, T, F>(s: String, ps: &'a ParseSess, f: F) -> T where
    F: FnOnce(&mut Parser<'a>) -> PResult<'a, T>,
{
    let mut p = string_to_parser(&ps, s);
    let x = panictry!(f(&mut p));
    p.abort_if_errors();
    x
}

/// Parse a string, return a crate.
pub fn string_to_crate (source_str : String) -> ast::Crate {
    let ps = ParseSess::new(FilePathMapping::empty());
    with_error_checking_parse(source_str, &ps, |p| {
        p.parse_crate_mod()
    })
}

/// Parse a string, return an expr
pub fn string_to_expr (source_str : String) -> P<ast::Expr> {
    let ps = ParseSess::new(FilePathMapping::empty());
    with_error_checking_parse(source_str, &ps, |p| {
        p.parse_expr()
    })
}

/// Parse a string, return an item
pub fn string_to_item (source_str : String) -> Option<P<ast::Item>> {
    let ps = ParseSess::new(FilePathMapping::empty());
    with_error_checking_parse(source_str, &ps, |p| {
        p.parse_item()
    })
}

/// Parse a string, return a pat. Uses "irrefutable"... which doesn't
/// (currently) affect parsing.
pub fn string_to_pat(source_str: String) -> P<ast::Pat> {
    let ps = ParseSess::new(FilePathMapping::empty());
    with_error_checking_parse(source_str, &ps, |p| {
        p.parse_pat()
    })
}

/// Convert a vector of strings to a vector of Ident's
pub fn strs_to_idents(ids: Vec<&str> ) -> Vec<Ident> {
    ids.iter().map(|u| Ident::from_str(*u)).collect()
}

/// Does the given string match the pattern? whitespace in the first string
/// may be deleted or replaced with other whitespace to match the pattern.
/// This function is relatively Unicode-ignorant; fortunately, the careful design
/// of UTF-8 mitigates this ignorance. It doesn't do NKF-normalization(?).
pub fn matches_codepattern(a : &str, b : &str) -> bool {
    let mut a_iter = a.chars().peekable();
    let mut b_iter = b.chars().peekable();

    loop {
        let (a, b) = match (a_iter.peek(), b_iter.peek()) {
            (None, None) => return true,
            (None, _) => return false,
            (Some(&a), None) => {
                if is_pattern_whitespace(a) {
                    break // trailing whitespace check is out of loop for borrowck
                } else {
                    return false
                }
            }
            (Some(&a), Some(&b)) => (a, b)
        };

        if is_pattern_whitespace(a) && is_pattern_whitespace(b) {
            // skip whitespace for a and b
            scan_for_non_ws_or_end(&mut a_iter);
            scan_for_non_ws_or_end(&mut b_iter);
        } else if is_pattern_whitespace(a) {
            // skip whitespace for a
            scan_for_non_ws_or_end(&mut a_iter);
        } else if a == b {
            a_iter.next();
            b_iter.next();
        } else {
            return false
        }
    }

    // check if a has *only* trailing whitespace
    a_iter.all(is_pattern_whitespace)
}

/// Advances the given peekable `Iterator` until it reaches a non-whitespace character
fn scan_for_non_ws_or_end<I: Iterator<Item= char>>(iter: &mut Peekable<I>) {
    while lexer::is_pattern_whitespace(iter.peek().cloned()) {
        iter.next();
    }
}

pub fn is_pattern_whitespace(c: char) -> bool {
    lexer::is_pattern_whitespace(Some(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eqmodws() {
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
        assert_eq!(matches_codepattern(" a   b","ab"),true);
    }

    #[test]
    fn pattern_whitespace() {
        assert_eq!(matches_codepattern("","\x0C"), false);
        assert_eq!(matches_codepattern("a b ","a   \u{0085}\n\t\r  b"),true);
        assert_eq!(matches_codepattern("a b","a   \u{0085}\n\t\r  b "),false);
    }

    #[test]
    fn non_pattern_whitespace() {
        // These have the property 'White_Space' but not 'Pattern_White_Space'
        assert_eq!(matches_codepattern("a b","a\u{2002}b"), false);
        assert_eq!(matches_codepattern("a   b","a\u{2002}b"), false);
        assert_eq!(matches_codepattern("\u{205F}a   b","ab"), false);
        assert_eq!(matches_codepattern("a  \u{3000}b","ab"), false);
    }
}
