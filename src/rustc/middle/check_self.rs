/*
  This module checks that within a class, "self" doesn't escape.
  That is, it rejects any class in which "self" occurs other than
  as the left-hand side of a field reference.
 */
import syntax::ast::*;
import syntax::visit::*;
import driver::session::session;
import std::map::hashmap;
import resolve::def_map;

fn check_crate(cx: ty::ctxt, crate: @crate) {
    visit_crate(*crate, cx, mk_vt(@{
        visit_item: bind check_item(_, _, _)
        with *default_visitor()
    }));
    cx.sess.abort_if_errors();
}

fn check_item(it: @item, &&cx: ty::ctxt, &&_v: vt<ty::ctxt>) {
    alt it.node {
      item_class(*) {
          visit_item(it, cx, check_self_visitor());
      }
      _ {}
    }
}

fn check_self_visitor() -> vt<ty::ctxt> {
    mk_vt(@{
        visit_expr: bind check_self_expr(_, _, _)
                with *default_visitor()
    })
}

fn check_self_expr(e: @expr, &&cx: ty::ctxt, &&v: vt<ty::ctxt>) {
   alt e.node {
     expr_field(@{node: expr_path(p),_},_,_) {
       // self is ok here; don't descend
     }
     expr_path(_) {
       alt cx.def_map.find(e.id) {
          some(def_self(_)) {
            cx.sess.span_err(e.span, "can't return self or store \
              it in a data structure");
          }
          _ {}
       }
     }
     _ { visit_expr(e, cx, v); }
  }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
