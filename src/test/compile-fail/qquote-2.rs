// xfail-pretty

use std;
use syntax;

import std::io::*;

import syntax::diagnostic;
import syntax::ast;
import syntax::codemap;
import syntax::parse::parser;
import syntax::print::*;

fn new_parse_sess() -> parser::parse_sess {
  fail;
}

iface fake_ext_ctxt {
    fn session() -> fake_session;
}

type fake_options = {cfg: ast::crate_cfg};

type fake_session = {opts: @fake_options,
                     parse_sess: parser::parse_sess};

impl of fake_ext_ctxt for fake_session {
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

