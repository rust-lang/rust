/*
 * The compiler code necessary to support the #env extension.  Eventually this
 * should all get sucked into either the compiler syntax extension plugin
 * interface.
 */

import util.common;

import std.Str;
import std.Vec;
import std.Option;
import std.GenericOS;

export expand_syntax_ext;

// FIXME: Need to thread parser through here to handle errors correctly
fn expand_syntax_ext(parser.parser p,
                     common.span sp,
                     vec[@ast.expr] args,
                     Option.t[str] body) -> @ast.expr {

    if (Vec.len[@ast.expr](args) != 1u) {
        p.err("malformed #env call");
    }

    // FIXME: if this was more thorough it would manufacture an
    // Option.t[str] rather than just an maybe-empty string.

    auto var = expr_to_str(p, args.(0));
    alt (GenericOS.getenv(var)) {
        case (Option.none[str]) {
            ret make_new_str(p, sp, "");
        }
        case (Option.some[str](?s)) {
            ret make_new_str(p, sp, s);
        }
    }
}

// FIXME: duplicate code copied from extfmt.

fn expr_to_str(parser.parser p,
               @ast.expr expr) -> str {
    alt (expr.node) {
        case (ast.expr_lit(?l, _)) {
            alt (l.node) {
                case (ast.lit_str(?s)) {
                    ret s;
                }
            }
        }
    }
    p.err("malformed #env call");
    fail;
}

fn make_new_lit(parser.parser p, common.span sp, ast.lit_ lit) -> @ast.expr {
    auto sp_lit = @rec(node=lit, span=sp);
    auto expr = ast.expr_lit(sp_lit, p.get_ann());
    ret @rec(node=expr, span=sp);
}

fn make_new_str(parser.parser p, common.span sp, str s) -> @ast.expr {
    auto lit = ast.lit_str(s);
    ret make_new_lit(p, sp, lit);
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
