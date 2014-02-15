// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty
// ignore-test

#[feature(quote)];

extern crate extra;
extern crate syntax;

use std::io::*;

use syntax::diagnostic;
use syntax::ast;
use syntax::codemap;
use syntax::codemap::span;
use syntax::parse;
use syntax::print::*;


trait fake_ext_ctxt {
    fn cfg() -> ast::CrateConfig;
    fn parse_sess() -> parse::parse_sess;
    fn call_site() -> span;
    fn ident_of(st: &str) -> ast::ident;
}

type fake_session = parse::parse_sess;

impl fake_ext_ctxt for fake_session {
    fn cfg() -> ast::CrateConfig { ~[] }
    fn parse_sess() -> parse::parse_sess { self }
    fn call_site() -> span {
        codemap::span {
            lo: codemap::BytePos(0),
            hi: codemap::BytePos(0),
            expn_info: None
        }
    }
    fn ident_of(st: &str) -> ast::ident {
        self.interner.intern(st)
    }
}

fn mk_ctxt() -> fake_ext_ctxt {
    parse::new_parse_sess(None) as fake_ext_ctxt
}

fn main() {
    let cx = mk_ctxt();

    let abc = quote_expr!(cx, 23);
    check_pp(ext_cx, abc,  pprust::print_expr, ~"23");


    let ty = quote_ty!(cx, int);
    check_pp(ext_cx, ty, pprust::print_type, ~"int");

    let item = quote_item!(cx, static x : int = 10;).get();
    check_pp(ext_cx, item, pprust::print_item, ~"static x: int = 10;");

    let stmt = quote_stmt!(cx, let x = 20;);
    check_pp(ext_cx, *stmt, pprust::print_stmt, ~"let x = 20;");

    let pat = quote_pat!(cx, Some(_));
    check_pp(ext_cx, pat, pprust::print_pat, ~"Some(_)");

}

fn check_pp<T>(cx: fake_ext_ctxt,
               expr: T, f: |pprust::ps, T|, expect: ~str) {
    let s = io::with_str_writer(|wr| {
        let pp = pprust::rust_printer(wr, cx.parse_sess().interner);
        f(pp, expr);
        pp::eof(pp.s);
    });
    stdout().write_line(s);
    if expect != ~"" {
        error!("expect: '%s', got: '%s'", expect, s);
        assert_eq!(s, expect);
    }
}
