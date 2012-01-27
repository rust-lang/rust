import driver::session;

import option::{none, some};

import syntax::ast::{crate, expr_, expr_mac, mac_invoc, mac_qq};
import syntax::fold::*;
import syntax::ext::base::*;
import syntax::ext::build::*;
import syntax::parse::parser::parse_expr_from_source_str;

import codemap::span;

fn expand_qquote(cx: ext_ctxt, sp: span, _e: @ast::expr) -> ast::expr_ {
    let str = codemap::span_to_snippet(sp, cx.session().parse_sess.cm);
    let session_call = bind mk_call_(cx,sp,
                                     mk_access(cx,sp,["ext_cx"], "session"),
                                     []);
    let call = mk_call(cx,sp,
                       ["syntax", "parse", "parser",
                        "parse_expr_from_source_str"],
                       [mk_str(cx,sp, "<anon>"),
                        mk_unary(cx,sp, ast::box(ast::imm),
                                 mk_str(cx,sp, str)),
                        mk_access_(cx,sp,
                                   mk_access_(cx,sp, session_call(), "opts"),
                                   "cfg"),
                        mk_access_(cx,sp, session_call(), "parse_sess")]
                      );
    ret call.node;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
