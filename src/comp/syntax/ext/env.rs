

/*
 * The compiler code necessary to support the #env extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */
import std::ivec;
import std::str;
import std::option;
import std::generic_os;
import base::*;
export expand_syntax_ext;

fn expand_syntax_ext(cx: &ext_ctxt, sp: codemap::span, args: &(@ast::expr)[],
                     body: option::t[str]) -> @ast::expr {
    if ivec::len[@ast::expr](args) != 1u {
        cx.span_fatal(sp, "malformed #env call");
    }
    // FIXME: if this was more thorough it would manufacture an
    // option::t[str] rather than just an maybe-empty string.

    let var = expr_to_str(cx, args.(0), "#env requires a string");
    alt generic_os::getenv(var) {
      option::none. { ret make_new_str(cx, sp, ""); }
      option::some(s) { ret make_new_str(cx, sp, s); }
    }
}

fn make_new_lit(cx: &ext_ctxt, sp: codemap::span, lit: ast::lit_) ->
   @ast::expr {
    let sp_lit = @{node: lit, span: sp};
    ret @{id: cx.next_id(), node: ast::expr_lit(sp_lit), span: sp};
}

fn make_new_str(cx: &ext_ctxt, sp: codemap::span, s: str) -> @ast::expr {
    ret make_new_lit(cx, sp, ast::lit_str(s, ast::sk_rc));
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
