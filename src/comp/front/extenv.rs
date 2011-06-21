

/*
 * The compiler code necessary to support the #env extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */
import util::common;
import std::str;
import std::vec;
import std::option;
import std::generic_os;
import ext::*;
export expand_syntax_ext;

fn expand_syntax_ext(&ext_ctxt cx, common::span sp, &vec[@ast::expr] args,
                     option::t[str] body) -> @ast::expr {
    if (vec::len[@ast::expr](args) != 1u) {
        cx.span_fatal(sp, "malformed #env call");
    }
    // FIXME: if this was more thorough it would manufacture an
    // option::t[str] rather than just an maybe-empty string.

    auto var = expr_to_str(cx, args.(0));
    alt (generic_os::getenv(var)) {
        case (option::none) { ret make_new_str(cx, sp, ""); }
        case (option::some(?s)) { ret make_new_str(cx, sp, s); }
    }
}


// FIXME: duplicate code copied from extfmt:
fn expr_to_str(&ext_ctxt cx, @ast::expr expr) -> str {
    alt (expr.node) {
        case (ast::expr_lit(?l)) {
            alt (l.node) {
                case (ast::lit_str(?s, _)) { ret s; }
                case (_) { cx.span_fatal(l.span, "malformed #env call"); }
            }
        }
        case (_) { cx.span_fatal(expr.span, "malformed #env call"); }
    }
}

fn make_new_lit(&ext_ctxt cx, common::span sp, ast::lit_ lit) -> @ast::expr {
    auto sp_lit = @rec(node=lit, span=sp);
    ret @rec(id=cx.next_id(), node=ast::expr_lit(sp_lit), span=sp);
}

fn make_new_str(&ext_ctxt cx, common::span sp, str s) -> @ast::expr {
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
