import syntax::ast;
import syntax::visit;
import option::some;
import syntax::print::pprust::expr_to_str;

export check_crate_fn_usage;

type fn_usage_ctx = {
    tcx: ty::ctxt,
    unsafe_fn_legal: bool,
    generic_bare_fn_legal: bool
};

fn fn_usage_expr(expr: @ast::expr,
                 ctx: fn_usage_ctx,
                 v: visit::vt<fn_usage_ctx>) {
    alt expr.node {
      ast::expr_path(path) {
        if !ctx.unsafe_fn_legal {
            alt ctx.tcx.def_map.find(expr.id) {
              some(ast::def_fn(_, ast::unsafe_fn.)) |
              some(ast::def_native_fn(_, ast::unsafe_fn.)) {
                log(error, ("expr=", expr_to_str(expr)));
                ctx.tcx.sess.span_fatal(
                    expr.span,
                    "unsafe functions can only be called");
              }

              _ {}
            }
        }
        if !ctx.generic_bare_fn_legal
            && ty::expr_has_ty_params(ctx.tcx, expr) {
            alt ty::struct(ctx.tcx, ty::expr_ty(ctx.tcx, expr)) {
              ty::ty_fn(ast::proto_bare., _, _, _, _) {
                ctx.tcx.sess.span_fatal(
                    expr.span,
                    "generic bare functions can only be called or bound");
              }
              _ { }
            }
        }
      }

      ast::expr_call(f, args, _) {
        let f_ctx = {unsafe_fn_legal: true,
                     generic_bare_fn_legal: true with ctx};
        v.visit_expr(f, f_ctx, v);

        let args_ctx = {unsafe_fn_legal: false,
                        generic_bare_fn_legal: false with ctx};
        visit::visit_exprs(args, args_ctx, v);
      }

      ast::expr_bind(f, args) {
        let f_ctx = {unsafe_fn_legal: false,
                     generic_bare_fn_legal: true with ctx};
        v.visit_expr(f, f_ctx, v);

        let args_ctx = {unsafe_fn_legal: false,
                        generic_bare_fn_legal: false with ctx};
        for arg in args {
            visit::visit_expr_opt(arg, args_ctx, v);
        }
      }

      _ {
        let subctx = {unsafe_fn_legal: false,
                      generic_bare_fn_legal: false with ctx};
        visit::visit_expr(expr, subctx, v);
      }
    }
}

fn check_crate_fn_usage(tcx: ty::ctxt, crate: @ast::crate) {
    let visit =
        visit::mk_vt(
            @{visit_expr: fn_usage_expr
                  with *visit::default_visitor()});
    let ctx = {
        tcx: tcx,
        unsafe_fn_legal: false,
        generic_bare_fn_legal: false
    };
    visit::visit_crate(*crate, ctx, visit);
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
