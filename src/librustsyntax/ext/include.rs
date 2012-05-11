/*
 * The compiler code necessary to support the #include and #include_str
 * extensions.  Eventually this should all get sucked into either the compiler
 * syntax extension plugin interface.
 */

import diagnostic::span_handler;
import base::*;
export str;

// FIXME: implement plain #include, restarting the parser on the included
// file. Currently only implement #include_str.

mod str {
    fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                         _body: ast::mac_body) -> @ast::expr {
        let arg = get_mac_arg(cx,sp,arg);
        let args: [@ast::expr] =
            alt arg.node {
              ast::expr_vec(elts, _) { elts }
              _ {
                cx.span_fatal(sp, "#include_str requires arguments \
                                   of the form `[...]`.")
              }
            };
        if vec::len::<@ast::expr>(args) != 1u {
            cx.span_fatal(sp, "malformed #include_str call");
        }
        let mut path = expr_to_str(cx, args[0], "#include_str requires \
                                                 a string");

        // NB: relative paths are resolved relative to the compilation unit
        if !path::path_is_absolute(path) {
            let cu = codemap::span_to_filename(sp, cx.codemap());
            let dir = path::dirname(cu);
            path = path::connect(dir, path);
        }

        alt io::read_whole_file_str(path) {
          result::ok(src) { ret make_new_str(cx, sp, src); }
          result::err(e) {
            cx.parse_sess().span_diagnostic.handler().fatal(e)
          }
        }
    }
}

fn make_new_str(cx: ext_ctxt, sp: codemap::span, s: str) -> @ast::expr {
    ret make_new_lit(cx, sp, ast::lit_str(s));
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
