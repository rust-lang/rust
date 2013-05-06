// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test Can't use syntax crate here

extern mod std;
extern mod syntax;

use std::io::*;

use syntax::diagnostic;
use syntax::ast;
use syntax::codemap;
use syntax::parse::parser;
use syntax::print::*;

trait fake_ext_ctxt {
    fn cfg() -> ast::crate_cfg;
    fn parse_sess() -> parse::parse_sess;
    fn call_site() -> span;
    fn ident_of(st: ~str) -> ast::ident;
}

type fake_session = parse::parse_sess;

impl fake_ext_ctxt for fake_session {
    fn cfg() -> ast::crate_cfg { ~[] }
    fn parse_sess() -> parse::parse_sess { self }
    fn call_site() -> span {
        codemap::span {
            lo: codemap::BytePos(0),
            hi: codemap::BytePos(0),
            expn_info: None
        }
    }
    fn ident_of(st: ~str) -> ast::ident {
        self.interner.intern(@st)
    }
}

fn mk_ctxt() -> fake_ext_ctxt {
    parse::new_parse_sess(None) as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();

    let stmt = quote_stmt!(let x int = 20;); //~ ERROR expected end-of-string
    check_pp(*stmt,  pprust::print_stmt, "");
}

fn check_pp<T>(expr: T, f: &fn(pprust::ps, T), expect: str) {
    fail!();
}
