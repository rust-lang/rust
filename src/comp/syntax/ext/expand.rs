import driver::session;

import option::{none, some};

import std::map::hashmap;
import vec;

import syntax::ast::{crate, expr_, expr_mac, mac_invoc, mac_qq};
import syntax::fold::*;
import syntax::ext::base::*;
import syntax::parse::parser::parse_expr_from_source_str;

import codemap::span;

fn expand_expr(exts: hashmap<str, syntax_extension>, cx: ext_ctxt,
               e: expr_, s: span, fld: ast_fold,
               orig: fn@(expr_, span, ast_fold) -> (expr_, span))
    -> (expr_, span)
{
    ret alt e {
          expr_mac(mac) {
            alt mac.node {
              mac_invoc(pth, args, body) {
                assert (vec::len(pth.node.idents) > 0u);
                let extname = pth.node.idents[0];
                alt exts.find(extname) {
                  none {
                    cx.span_fatal(pth.span,
                                  #fmt["macro undefined: '%s'", extname])
                  }
                  some(normal(ext)) {
                    let expanded = ext(cx, pth.span, args, body);

                    cx.bt_push(mac.span);
                    //keep going, outside-in
                    let fully_expanded = fld.fold_expr(expanded).node;
                    cx.bt_pop();

                    (fully_expanded, s)
                  }
                  some(macro_defining(ext)) {
                    let named_extension = ext(cx, pth.span, args, body);
                    exts.insert(named_extension.ident, named_extension.ext);
                    (ast::expr_rec([], none), s)
                  }
                }
              }
              mac_qq(sp, exp) { (expand_qquote(cx, sp, exp), s) }
              _ { cx.span_bug(mac.span, "naked syntactic bit") }
            }
          }
          _ { orig(e, s, fld) }
        };
}

fn expand_qquote(cx: ext_ctxt, sp: span, e: @ast::expr) -> ast::expr_ {
    import syntax::ext::build::*;
    let str = codemap::span_to_snippet(sp, cx.session().parse_sess.cm);
    let expr = make_new_str(cx, e.span, str);
    ret expr.node;
}

// FIXME: this is a terrible kludge to inject some macros into the default
// compilation environment. When the macro-definition system is substantially
// more mature, these should move from here, into a compiled part of libcore
// at very least.

fn core_macros() -> str {
    ret
"{
    #macro([#error[f, ...], log(core::error, #fmt[f, ...])]);
    #macro([#warn[f, ...], log(core::warn, #fmt[f, ...])]);
    #macro([#info[f, ...], log(core::info, #fmt[f, ...])]);
    #macro([#debug[f, ...], log(core::debug, #fmt[f, ...])]);
}";
}

fn expand_crate(sess: session::session, c: @crate) -> @crate {
    let exts = syntax_expander_table();
    let afp = default_ast_fold();
    let cx: ext_ctxt = mk_ctxt(sess);
    let f_pre =
        {fold_expr: bind expand_expr(exts, cx, _, _, _, afp.fold_expr)
            with *afp};
    let f = make_fold(f_pre);
    let cm = parse_expr_from_source_str("<anon>", @core_macros(),
                                        sess.opts.cfg,
                                        sess.parse_sess);

    // This is run for its side-effects on the expander env,
    // as it registers all the core macros as expanders.
    f.fold_expr(cm);

    let res = @f.fold_crate(*c);
    ret res;
}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
