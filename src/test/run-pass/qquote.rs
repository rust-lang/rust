// xfail-pretty

use std;
use rustc;

import rustc::*;
import io::*;

import rustc::driver::diagnostic;
import rustc::syntax::ast;
import rustc::syntax::codemap;
import rustc::syntax::parse::parser;
import rustc::syntax::print::*;

fn new_parse_sess() -> parser::parse_sess {
    let cm = codemap::new_codemap();
    let handler = diagnostic::mk_handler(option::none);
    let sess = @{
        cm: cm,
        mutable next_id: 1,
        span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        mutable chpos: 0u,
        mutable byte_pos: 0u
    };
    ret sess;
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
    let opts : fake_options = {cfg: []};
    {opts: @opts, parse_sess: new_parse_sess()} as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();

    let abc = #ast{23};
    check_pp(abc,  pprust::print_expr, "23");

    let expr3 = #ast{2 - $(abc) + 7};
    check_pp(expr3,  pprust::print_expr, "2 - 23 + 7");

    let expr4 = #ast{2 - $(#ast{3}) + 9};
    check_pp(expr4,  pprust::print_expr, "2 - 3 + 9");

    let ty = #ast(ty){int};
    check_pp(ty, pprust::print_type, "int");

    let ty2 = #ast(ty){option<$(ty)>};
    check_pp(ty2, pprust::print_type, "option<int>");

    let item = #ast(item){const x : int = 10;};
    check_pp(item, pprust::print_item, "const x: int = 10;");

    let item2: @ast::item = #ast(item){const x : int = $(abc);};
    check_pp(item2, pprust::print_item, "const x: int = 23;");

    let stmt = #ast(stmt){let x = 20;};
    check_pp(*stmt, pprust::print_stmt, "let x = 20;");

    let stmt2 = #ast(stmt){let x : $(ty) = $(abc);};
    check_pp(*stmt2, pprust::print_stmt, "let x: int = 23;");

    let pat = #ast(pat){some(_)};
    check_pp(pat, pprust::print_pat, "some(_)");

    // issue #1785
    let x = #ast{1};
    let test1 = #ast{1+$(x)};
    check_pp(test1, pprust::print_expr, "1 + 1");

    let test2 = #ast{$(x)+1};
    check_pp(test2, pprust::print_expr, "1 + 1");

    let y = #ast{2};
    let test3 = #ast{$(x) + $(y)};
    check_pp(test3, pprust::print_expr, "1 + 2");

    let crate = #ast(crate) { fn a() { } };
    check_pp(crate, pprust::print_crate_, "fn a() { }\n");

    // issue #1926
    let s = #ast(expr){__s};
    let e = #ast(expr){__e};
    let call = #ast(expr){$(s).foo {|__e| $(e)}};
    check_pp(call, pprust::print_expr, "__s.foo {|__e| __e }")
}

fn check_pp<T>(expr: T, f: fn(pprust::ps, T), expect: str) {
    let buf = mem_buffer();
    let pp = pprust::rust_printer(buf as io::writer);
    f(pp, expr);
    pp::eof(pp.s);
    let str = mem_buffer_str(buf);
    stdout().write_line(str);
    if expect != "" {
        #error("expect: '%s', got: '%s'", expect, str);
        assert str == expect;
    }
}

