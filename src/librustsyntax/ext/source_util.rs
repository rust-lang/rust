import base::*;
import ast;
import codemap::span;
import print::pprust;


/* #line(): expands to the current line number */
fn expand_line(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
               _body: ast::mac_body) -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "line");
    let loc = codemap::lookup_char_pos(cx.codemap(), sp.lo);
    ret make_new_lit(cx, sp, ast::lit_uint(loc.line as u64, ast::ty_u));
}

/* #col(): expands to the current column number */
fn expand_col(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
              _body: ast::mac_body) -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "col");
    let loc = codemap::lookup_char_pos(cx.codemap(), sp.lo);
    ret make_new_lit(cx, sp, ast::lit_uint(loc.col as u64, ast::ty_u));
}

/* #file(): expands to the current filename */
/* The filemap (`loc.file`) contains a bunch more information we could spit
 * out if we wanted. */
fn expand_file(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
               _body: ast::mac_body) -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "file");
    let loc = codemap::lookup_char_pos(cx.codemap(), sp.lo);
    ret make_new_lit(cx, sp, ast::lit_str(loc.file.name));
}

fn expand_stringify(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                    _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx, sp, arg, 1u, option::some(1u), "stringify");
    ret make_new_lit(cx, sp, ast::lit_str(pprust::expr_to_str(args[0])));
}

fn expand_include(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                  _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx, sp, arg, 1u, option::some(1u), "include");
    let loc = codemap::lookup_char_pos(cx.codemap(), sp.lo);
    let path = path::connect(path::dirname(loc.file.name),
        expr_to_str(cx, args[0], "#include requires a string literal"));
    let p = parse::new_parser_from_file(cx.parse_sess(), cx.cfg(), path,
                                        parse::parser::SOURCE_FILE);
    ret parse::parser::parse_expr(p)
}
