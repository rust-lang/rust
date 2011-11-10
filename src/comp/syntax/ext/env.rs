
/*
 * The compiler code necessary to support the #env extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */
import std::{vec, option, generic_os};
import base::*;
export expand_syntax_ext;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: @ast::expr,
                     _body: option::t<str>) -> @ast::expr {
    let args: [@ast::expr] =
        alt arg.node {
          ast::expr_vec(elts, _) { elts }
          _ {
            cx.span_fatal(sp, "#env requires arguments of the form `[...]`.")
          }
        };
    if vec::len::<@ast::expr>(args) != 1u {
        cx.span_fatal(sp, "malformed #env call");
    }
    // FIXME: if this was more thorough it would manufacture an
    // option::t<str> rather than just an maybe-empty string.

    let var = expr_to_str(cx, args[0], "#env requires a string");
    alt generic_os::getenv(var) {
      option::none. { ret make_new_str(cx, sp, ""); }
      option::some(s) { ret make_new_str(cx, sp, s); }
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
