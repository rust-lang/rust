
/*
 * The compiler code necessary to support the #env extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */
import base::*;
import build::mk_uniq_str;
export expand_syntax_ext;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args(cx, sp, arg, 1u, option::Some(1u), ~"env");

    // FIXME (#2248): if this was more thorough it would manufacture an
    // Option<str> rather than just an maybe-empty string.

    let var = expr_to_str(cx, args[0], ~"#env requires a string");
    match os::getenv(var) {
      option::None => return mk_uniq_str(cx, sp, ~""),
      option::Some(s) => return mk_uniq_str(cx, sp, s)
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
