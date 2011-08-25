import driver::session;

import std::option::none;
import std::option::some;

import std::map::hashmap;
import std::vec;
import std::istr;

import syntax::ast::crate;
import syntax::ast::expr_;
import syntax::ast::expr_mac;
import syntax::ast::mac_invoc;
import syntax::fold::*;
import syntax::ext::base::*;


fn expand_expr(exts: &hashmap<istr, syntax_extension>, cx: &ext_ctxt,
               e: &expr_, fld: ast_fold, orig: &fn(&expr_, ast_fold) -> expr_)
   -> expr_ {
    ret alt e {
          expr_mac(mac) {
            alt mac.node {
              mac_invoc(pth, args, body) {
                assert (vec::len(pth.node.idents) > 0u);
                let extname = pth.node.idents[0];
                alt exts.find(istr::from_estr(extname)) {
                  none. {
                    cx.span_fatal(pth.span,
                                  #fmt["macro undefined: '%s'", extname])
                  }
                  some(normal(ext)) {
                    let expanded = ext(cx, pth.span, args, body);

                    cx.bt_push(mac.span);
                    //keep going, outside-in
                    let fully_expanded = fld.fold_expr(expanded).node;
                    cx.bt_pop();

                    fully_expanded
                  }
                  some(macro_defining(ext)) {
                    let named_extension = ext(cx, pth.span, args, body);
                    exts.insert(
                        istr::from_estr(named_extension.ident),
                        named_extension.ext);
                    ast::expr_rec([], none)
                  }
                }
              }
              _ { cx.span_bug(mac.span, "naked syntactic bit") }
            }
          }
          _ { orig(e, fld) }
        };
}

fn expand_crate(sess: &session::session, c: &@crate) -> @crate {
    let exts = syntax_expander_table();
    let afp = default_ast_fold();
    let cx: ext_ctxt = mk_ctxt(sess);
    let f_pre =
        {fold_expr: bind expand_expr(exts, cx, _, _, afp.fold_expr)
            with *afp};
    let f = make_fold(f_pre);
    let res = @f.fold_crate(*c);
    dummy_out(f); //temporary: kill circular reference
    ret res;

}
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
