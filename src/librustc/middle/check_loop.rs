use syntax::ast::*;
use syntax::visit;

type ctx = {in_loop: bool, can_ret: bool};

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    visit::visit_crate(*crate, {in_loop: false,can_ret: true}, visit::mk_vt(@{
        visit_item: |i, _cx, v| {
            visit::visit_item(i, {in_loop: false, can_ret: true}, v);
        },
        visit_expr: |e: @expr, cx: ctx, v: visit::vt<ctx>| {
            match e.node {
              expr_while(e, b) => {
                v.visit_expr(e, cx, v);
                v.visit_block(b, {in_loop: true,.. cx}, v);
              }
              expr_loop(b, _) => {
                v.visit_block(b, {in_loop: true,.. cx}, v);
              }
              expr_fn(_, _, _, _) => {
                visit::visit_expr(e, {in_loop: false, can_ret: true}, v);
              }
              expr_fn_block(_, b, _) => {
                v.visit_block(b, {in_loop: false, can_ret: false}, v);
              }
              expr_loop_body(@{node: expr_fn_block(_, b, _), _}) => {
                let proto = ty::ty_fn_proto(ty::expr_ty(tcx, e));
                let blk = (proto == ProtoBorrowed);
                v.visit_block(b, {in_loop: true, can_ret: blk}, v);
              }
              expr_break(_) => {
                if !cx.in_loop {
                    tcx.sess.span_err(e.span, ~"`break` outside of loop");
                }
              }
              expr_again(_) => {
                if !cx.in_loop {
                    tcx.sess.span_err(e.span, ~"`again` outside of loop");
                }
              }
              expr_ret(oe) => {
                if !cx.can_ret {
                    tcx.sess.span_err(e.span, ~"`return` in block function");
                }
                visit::visit_expr_opt(oe, cx, v);
              }
              _ => visit::visit_expr(e, cx, v)
            }
        },
        .. *visit::default_visitor()
    }));
}
