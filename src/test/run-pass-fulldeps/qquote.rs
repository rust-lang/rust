// xfail-pretty

use std;
use syntax;

import io::*;

import syntax::diagnostic;
import syntax::ast;
import syntax::codemap;
import syntax::parse;
import syntax::print::*;

trait fake_ext_ctxt {
    fn cfg() -> ast::crate_cfg;
    fn parse_sess() -> parse::parse_sess;
}

type fake_session = parse::parse_sess;

impl fake_session: fake_ext_ctxt {
    fn cfg() -> ast::crate_cfg { ~[] }
    fn parse_sess() -> parse::parse_sess { self }
}

fn mk_ctxt() -> fake_ext_ctxt {
    parse::new_parse_sess(none) as fake_ext_ctxt
}


fn main() {
    let ext_cx = mk_ctxt();

    let abc = #ast{23};
    check_pp(ext_cx, abc,  pprust::print_expr, ~"23");

    let expr3 = #ast{2 - $(abc) + 7};
    check_pp(ext_cx, expr3,  pprust::print_expr, ~"2 - 23 + 7");

    let expr4 = #ast{2 - $(#ast{3}) + 9};
    check_pp(ext_cx, expr4,  pprust::print_expr, ~"2 - 3 + 9");

    let ty = #ast[ty]{int};
    check_pp(ext_cx, ty, pprust::print_type, ~"int");

    let ty2 = #ast[ty]{option<$(ty)>};
    check_pp(ext_cx, ty2, pprust::print_type, ~"option<int>");

    let item = #ast[item]{const x : int = 10;};
    check_pp(ext_cx, item, pprust::print_item, ~"const x: int = 10;");

    let item2: @ast::item = #ast[item]{const x : int = $(abc);};
    check_pp(ext_cx, item2, pprust::print_item, ~"const x: int = 23;");

    let stmt = #ast[stmt]{let x = 20;};
    check_pp(ext_cx, *stmt, pprust::print_stmt, ~"let x = 20;");

    let stmt2 = #ast[stmt]{let x : $(ty) = $(abc);};
    check_pp(ext_cx, *stmt2, pprust::print_stmt, ~"let x: int = 23;");

    let pat = #ast[pat]{some(_)};
    check_pp(ext_cx, pat, pprust::print_pat, ~"some(_)");

    // issue #1785
    let x = #ast{1};
    let test1 = #ast{1+$(x)};
    check_pp(ext_cx, test1, pprust::print_expr, ~"1 + 1");

    let test2 = #ast{$(x)+1};
    check_pp(ext_cx, test2, pprust::print_expr, ~"1 + 1");

    let y = #ast{2};
    let test3 = #ast{$(x) + $(y)};
    check_pp(ext_cx, test3, pprust::print_expr, ~"1 + 2");

    let crate = #ast[crate] { fn a() { } };
    check_pp(ext_cx, crate, pprust::print_crate_, ~"fn a() { }\n");

    // issue #1926
    let s = #ast[expr]{__s};
    let e = #ast[expr]{__e};
    let call = #ast[expr]{$(s).foo(|__e| $(e) )};
    check_pp(ext_cx, call, pprust::print_expr, ~"__s.foo(|__e| __e)")
}

fn check_pp<T>(cx: fake_ext_ctxt,
               expr: T, f: fn(pprust::ps, T), expect: ~str) {
    let buf = mem_buffer();
    let pp = pprust::rust_printer(buf as io::Writer,cx.parse_sess().interner);
    f(pp, expr);
    pp::eof(pp.s);
    let str = mem_buffer_str(buf);
    stdout().write_line(str);
    if expect != ~"" {
        error!{"expect: '%s', got: '%s'", expect, str};
        assert str == expect;
    }
}

