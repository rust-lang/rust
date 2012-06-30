import base::*;
import ast;
import codemap::span;
import print::pprust;
import build::{mk_lit,mk_uniq_vec_e};

export expand_line;
export expand_col;
export expand_file;
export expand_stringify;
export expand_mod;
export expand_include;
export expand_include_str;
export expand_include_bin;

/* #line(): expands to the current line number */
fn expand_line(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
               _body: ast::mac_body) -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "line");
    let loc = codemap::lookup_char_pos(cx.codemap(), sp.lo);
    ret mk_lit(cx, sp, ast::lit_uint(loc.line as u64, ast::ty_u));
}

/* #col(): expands to the current column number */
fn expand_col(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
              _body: ast::mac_body) -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "col");
    let loc = codemap::lookup_char_pos(cx.codemap(), sp.lo);
    ret mk_lit(cx, sp, ast::lit_uint(loc.col as u64, ast::ty_u));
}

/* #file(): expands to the current filename */
/* The filemap (`loc.file`) contains a bunch more information we could spit
 * out if we wanted. */
fn expand_file(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
               _body: ast::mac_body) -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "file");
    let { file: @{ name: filename, _ }, _ } =
        codemap::lookup_char_pos(cx.codemap(), sp.lo);
    ret mk_lit(cx, sp, ast::lit_str(@filename));
}

fn expand_stringify(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                    _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx, sp, arg, 1u, option::some(1u), "stringify");
    ret mk_lit(cx, sp, ast::lit_str(@pprust::expr_to_str(args[0])));
}

fn expand_mod(cx: ext_ctxt, sp: span, arg: ast::mac_arg, _body: ast::mac_body)
    -> @ast::expr {
    get_mac_args(cx, sp, arg, 0u, option::some(0u), "file");
    ret mk_lit(cx, sp, ast::lit_str(
        @str::connect(cx.mod_path().map(|x|*x), "::")));
}

fn expand_include(cx: ext_ctxt, sp: span, arg: ast::mac_arg,
                  _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx, sp, arg, 1u, option::some(1u), "include");
    let file = expr_to_str(cx, args[0], "#include_str requires a string");
    let p = parse::new_parser_from_file(cx.parse_sess(), cx.cfg(),
                                        res_rel_file(cx, sp, file),
                                        parse::parser::SOURCE_FILE);
    ret p.parse_expr();
}

fn expand_include_str(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                      _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx,sp,arg,1u,option::some(1u),"include_str");

    let file = expr_to_str(cx, args[0], "#include_str requires a string");

    let res = io::read_whole_file_str(res_rel_file(cx, sp, file));
    alt res {
      result::ok(_) { /* Continue. */ }
      result::err(e) {
        cx.parse_sess().span_diagnostic.handler().fatal(e);
      }
    }

    ret mk_lit(cx, sp, ast::lit_str(@result::unwrap(res)));
}

fn expand_include_bin(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                      _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx,sp,arg,1u,option::some(1u),"include_bin");

    let file = expr_to_str(cx, args[0], "#include_bin requires a string");

    alt io::read_whole_file(res_rel_file(cx, sp, file)) {
      result::ok(src) {
        let u8_exprs = vec::map(src, |char: u8| {
            mk_lit(cx, sp, ast::lit_uint(char as u64, ast::ty_u8))
        });
        ret mk_uniq_vec_e(cx, sp, u8_exprs);
      }
      result::err(e) {
        cx.parse_sess().span_diagnostic.handler().fatal(e)
      }
    }
}

fn res_rel_file(cx: ext_ctxt, sp: codemap::span, +arg: path) -> path {
    // NB: relative paths are resolved relative to the compilation unit
    if !path::path_is_absolute(arg) {
        let cu = codemap::span_to_filename(sp, cx.codemap());
        let dir = path::dirname(cu);
        ret path::connect(dir, arg);
    } else {
        ret arg;
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
