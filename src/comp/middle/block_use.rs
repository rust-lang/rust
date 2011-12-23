import syntax::visit;
import syntax::ast::*;

type ctx = {tcx: ty::ctxt, mutable allow_block: bool};

fn check_crate(tcx: ty::ctxt, crate: @crate) {
    let cx = {tcx: tcx, mutable allow_block: false};
    let v = visit::mk_vt(@{visit_expr: visit_expr
                           with *visit::default_visitor()});
    visit::visit_crate(*crate, cx, v);
}

fn visit_expr(ex: @expr, cx: ctx, v: visit::vt<ctx>) {
    if !cx.allow_block {
        alt ty::struct(cx.tcx, ty::expr_ty(cx.tcx, ex)) {
          ty::ty_fn({proto: proto_block., _}) {
            cx.tcx.sess.span_err(ex.span, "expressions with block type \
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
        let i = 0u;
        for arg_t in ty::ty_fn_args(cx.tcx, ty::expr_ty(cx.tcx, f)) {
            cx.allow_block = arg_t.mode == by_ref;
            v.visit_expr(args[i], cx, v);
            i += 1u;
        }
      }
      _ {
        cx.allow_block = false;
        visit::visit_expr(ex, cx, v);
      }
    }
    cx.allow_block = outer;
}
