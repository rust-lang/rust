import driver::session;

import option::{none, some};

import syntax::ast::{crate, expr_, expr_mac, mac_invoc, mac_qq, mac_var};
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

fn replace(e: @ast::expr, repls: [@ast::expr]) -> @ast::expr {
    let aft = default_ast_fold();
    let f_pre = {fold_expr: bind replace_expr(repls, _, _, _,
                                              aft.fold_expr)
                 with *aft};
    let f = make_fold(f_pre);
    ret f.fold_expr(e);
}

fn replace_expr(repls: [@ast::expr],
                e: ast::expr_, s: span, fld: ast_fold,
                orig: fn@(ast::expr_, span, ast_fold)->(ast::expr_, span))
    -> (ast::expr_, span)
{
    // note: nested enum matching will be really nice here so I can jusy say
    //       expr_mac(mac_var(i))
    alt e {
      expr_mac({node: mac_var(i), _}) {let r = repls[i]; (r.node, r.span)}
      _ {orig(e,s,fld)}
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
