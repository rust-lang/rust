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

extern mod extra;
extern mod syntax;

use io::*;

use syntax::diagnostic;
use syntax::ast;
use syntax::codemap;
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
    check_pp(abc,  pprust::print_expr, "23");

    let expr3 = quote_expr!(cx, 2 - $abcd + 7); //~ ERROR unresolved name: abcd
    check_pp(expr3,  pprust::print_expr, "2 - 23 + 7");
}

fn check_pp<T>(expr: T, f: &fn(pprust::ps, T), expect: str) {
    fail!();
}
