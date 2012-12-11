// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


extern mod std;
use syntax;

use std::io::*;

use syntax::diagnostic;
use syntax::ast;
use syntax::codemap;
use syntax::parse::parser;
use syntax::print::*;

fn new_parse_sess() -> parser::parse_sess {
  fail;
}

trait fake_ext_ctxt {
    fn session() -> fake_session;
}

type fake_options = {cfg: ast::crate_cfg};

type fake_session = {opts: @fake_options,
                     parse_sess: parser::parse_sess};

impl fake_session: fake_ext_ctxt {
    fn session() -> fake_session {self}
}

fn mk_ctxt() -> fake_ext_ctxt {
    let opts : fake_options = {cfg: ~[]};
    {opts: @opts, parse_sess: new_parse_sess()} as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();

    let stmt = #ast[stmt]{let x int = 20;}; //~ ERROR expected end-of-string
    check_pp(*stmt,  pprust::print_stmt, "");
}

fn check_pp<T>(expr: T, f: fn(pprust::ps, T), expect: str) {
    fail;
}

