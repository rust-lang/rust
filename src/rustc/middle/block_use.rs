import syntax::visit;
import syntax::ast::*;
import driver::session::session;

type ctx = {tcx: ty::ctxt, mut allow_block: bool};

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    let cx = {tcx: tcx, mut allow_block: false};
    let v = visit::mk_vt(@{visit_expr: visit_expr
                           with *visit::default_visitor()});
    visit::visit_crate(*crate, cx, v);
}

fn visit_expr(ex: @expr, cx: ctx, v: visit::vt<ctx>) {
    if !cx.allow_block {
        alt ty::get(ty::expr_ty(cx.tcx, ex)).struct {
          ty::ty_fn({proto: p, _}) if is_blockish(p) {
            cx.tcx.sess.span_err(ex.span,
               "expressions with stack closure type \
                can only appear in callee or (by-ref) argument position");
          }
          _ {}
        }
    }
    let outer = cx.allow_block;
    alt ex.node {
      expr_call(f, args, _) {
        cx.allow_block = true;
        v.visit_expr(f, cx, v);
        let mut i = 0u;
        for ty::ty_fn_args(ty::expr_ty(cx.tcx, f)).each {|arg_t|
            cx.allow_block = (ty::arg_mode(cx.tcx, arg_t) == by_ref);
            v.visit_expr(args[i], cx, v);
            i += 1u;
        }
      }
      expr_loop_body(body) | expr_do_body(body) {
        cx.allow_block = true;
        v.visit_expr(body, cx, v);
      }
      _ {
        cx.allow_block = false;
        visit::visit_expr(ex, cx, v);
      }
    }
    cx.allow_block = outer;
}
