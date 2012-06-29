// xfail-pretty

use std;
use syntax;

import io::*;

import syntax::diagnostic;
import syntax::ast;
import syntax::codemap;
import syntax::parse;
import syntax::print::*;

fn new_parse_sess() -> parse::parse_sess {
  fail;
}

iface fake_ext_ctxt {
    fn session() -> fake_session;
}

type fake_options = {cfg: ast::crate_cfg};

type fake_session = {opts: @fake_options,
                     parse_sess: parse::parse_sess};

impl of fake_ext_ctxt for fake_session {
    fn session() -> fake_session {self}
}

fn mk_ctxt() -> fake_ext_ctxt {
    let opts : fake_options = {cfg: ~[]};
    {opts: @opts, parse_sess: new_parse_sess()} as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();

    let abc = #ast{23};
    check_pp(abc,  pprust::print_expr, "23");

    let expr3 = #ast{2 - $(abcd) + 7}; //! ERROR unresolved name: abcd
    check_pp(expr3,  pprust::print_expr, "2 - 23 + 7");
}

fn check_pp<T>(expr: T, f: fn(pprust::ps, T), expect: str) {
    fail;
}

