import syntax::ast::*;
import syntax::visit;

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    let v =
        @{visit_item: bind check_item(tcx, _, _, _)
             with *visit::default_visitor::<()>()};
    visit::visit_crate(*crate, (), visit::mk_vt(v));
    tcx.sess.abort_if_errors();
}

fn check_item(tcx: ty::ctxt, it: @item, &&s: (), v: visit::vt<()>) {
    visit::visit_item(it, s, v);
    alt it.node {
      item_const(_ /* ty */, ex) {
         let v =
             @{visit_expr: bind check_const_expr(tcx, _, _, _)
                  with *visit::default_visitor::<()>()};
         check_const_expr(tcx, ex, (), visit::mk_vt(v));
       }
       _ { }
    }
}

fn check_const_expr(tcx: ty::ctxt, ex: @expr, &&s: (), v: visit::vt<()>) {
    visit::visit_expr(ex, s, v);
    alt ex.node {
      expr_lit(_) { }
      expr_binary(_, _, _) { /* subexps covered by visit */ }
      expr_unary(u, _) {
        alt u {
          box(_)  |
          uniq(_) |
          deref.  {
            tcx.sess.span_err(ex.span,
                              "disallowed operator in constant expression");
          }
          _ { }
        }
      }
      _ { tcx.sess.span_err(ex.span,
            "constant contains unimplemented expression type"); }
    }
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
