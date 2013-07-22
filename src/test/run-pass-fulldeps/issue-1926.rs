// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// fails pretty printing for some reason
use syntax;
use syntax::diagnostic;
use syntax::ast;
use syntax::codemap;
use syntax::print::pprust;
use syntax::parse::parser;

fn new_parse_sess() -> parser::parse_sess {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(option::none);
    let sess = @mut {
        cm: cm,
        next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        chpos: 0u,
        byte_pos: 0u
    };
    return sess;
}

trait fake_ext_ctxt {
    fn session() -> fake_session;
    fn cfg() -> ast::Crate_cfg;
    fn parse_sess() -> parser::parse_sess;
}

type fake_options = {cfg: ast::Crate_cfg};

type fake_session = {opts: @fake_options,
                     parse_sess: parser::parse_sess};

impl of fake_ext_ctxt for fake_session {
    fn session() -> fake_session {self}
    fn cfg() -> ast::Crate_cfg { self.opts.cfg }
    fn parse_sess() -> parser::parse_sess { self.parse_sess }
}

fn mk_ctxt() -> fake_ext_ctxt {
    let opts : fake_options = {cfg: ~[]};
    {opts: @opts, parse_sess: new_parse_sess()} as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();
    let s = quote_expr!(__s);
    let e = quote_expr!(__e);
    let f = quote_expr!($s.foo {|__e| $e});
    log(error, pprust::expr_to_str(f));
}
