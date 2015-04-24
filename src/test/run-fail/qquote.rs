// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cross-compile

// error-pattern:expected identifier, found keyword `let`

#![feature(quote, rustc_private)]

extern crate syntax;

use syntax::ast;
use syntax::codemap;
use syntax::parse;
use syntax::print::pprust;

trait FakeExtCtxt {
    fn call_site(&self) -> codemap::Span;
    fn cfg(&self) -> ast::CrateConfig;
    fn ident_of(&self, st: &str) -> ast::Ident;
    fn name_of(&self, st: &str) -> ast::Name;
    fn parse_sess(&self) -> &parse::ParseSess;
}

impl FakeExtCtxt for parse::ParseSess {
    fn call_site(&self) -> codemap::Span {
        codemap::Span {
            lo: codemap::BytePos(0),
            hi: codemap::BytePos(0),
            expn_id: codemap::NO_EXPANSION,
        }
    }
    fn cfg(&self) -> ast::CrateConfig { Vec::new() }
    fn ident_of(&self, st: &str) -> ast::Ident {
        parse::token::str_to_ident(st)
    }
    fn name_of(&self, st: &str) -> ast::Name {
        parse::token::intern(st)
    }
    fn parse_sess(&self) -> &parse::ParseSess { self }
}

fn main() {
    let cx = parse::new_parse_sess();

    assert_eq!(pprust::expr_to_string(&*quote_expr!(&cx, 23)), "23");

    let expr = quote_expr!(&cx, let x isize = 20;);
    assert_eq!(pprust::expr_to_string(&*expr), "let x isize = 20;");
}
