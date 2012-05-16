/*

The region check is a final pass that runs over the AST after we have
inferred the type constraints but before we have actually finalized
the types.  It's purpose is to embed some final region constraints.
The reason that this is not done earlier is that sometimes we don't
know whether a given type will be a region pointer or not until this
phase.

In particular, we ensure that, if the type of an expression or
variable is `&r.T`, then the expression or variable must occur within
the region scope `r`.

*/

import util::ppaux;
import syntax::print::pprust;

fn regionck_expr(fcx: @fn_ctxt, e: @ast::expr) {
    let v = regionck_visitor(fcx);
    v.visit_expr(e, fcx, v);
}

fn regionck_fn(fcx: @fn_ctxt,
               _decl: ast::fn_decl,
               blk: ast::blk) {
    let v = regionck_visitor(fcx);
    v.visit_block(blk, fcx, v);
}

type rvt = visit::vt<@fn_ctxt>;

fn regionck_visitor(_fcx: @fn_ctxt) -> rvt {
    visit::mk_vt(@{visit_item: visit_item,
                   visit_stmt: visit_stmt,
                   visit_expr: visit_expr,
                   visit_block: visit_block,
                   visit_pat: visit_pat,
                   visit_local: visit_local
                   with *visit::default_visitor()})
}

fn visit_item(_item: @ast::item, &&_fcx: @fn_ctxt, _v: rvt) {
    // Ignore items
}

fn visit_local(l: @ast::local, &&fcx: @fn_ctxt, v: rvt) {
    visit::visit_local(l, fcx, v);
}

fn visit_pat(p: @ast::pat, &&fcx: @fn_ctxt, v: rvt) {
    visit::visit_pat(p, fcx, v);
}

fn visit_block(b: ast::blk, &&fcx: @fn_ctxt, v: rvt) {
    visit::visit_block(b, fcx, v);
}

fn visit_expr(e: @ast::expr, &&fcx: @fn_ctxt, v: rvt) {
    #debug["visit_expr(e=%s)", pprust::expr_to_str(e)];

    visit_ty(fcx.expr_ty(e), e.id, e.span, fcx);
    visit::visit_expr(e, fcx, v);
}

fn visit_stmt(s: @ast::stmt, &&fcx: @fn_ctxt, v: rvt) {
    visit::visit_stmt(s, fcx, v);
}

fn visit_ty(ty: ty::t,
            id: ast::node_id,
            span: span,
            fcx: @fn_ctxt) {

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty = alt infer::resolve_deep(fcx.infcx, ty, false) {
      result::err(_) { ret; }
      result::ok(ty) { ty }
    };

    // find the region where this expr evaluation is taking place
    let tcx = fcx.ccx.tcx;
    let encl_region = ty::encl_region(tcx, id);

    #debug["visit_ty(ty=%s, id=%d, encl_region=%s)",
           ppaux::ty_to_str(tcx, ty),
           id,
           ppaux::region_to_str(tcx, encl_region)];

    // Otherwise, look at the type and see if it is a region pointer.
    if !ty::type_has_regions(ty) { ret; }
    ty::walk_regions_and_ty(
        tcx, ty,
        { |r| constrain_region(fcx, encl_region, span, r); },
        { |t| ty::type_has_regions(t) });

    fn constrain_region(fcx: @fn_ctxt,
                        encl_region: ty::region,
                        span: span,
                        region: ty::region) {
        let tcx = fcx.ccx.tcx;

        #debug["constrain_region(encl_region=%s, region=%s)",
               ppaux::region_to_str(tcx, encl_region),
               ppaux::region_to_str(tcx, region)];

        alt region {
          ty::re_bound(_) {
            // a bound region is one which appears inside an fn type.
            // (e.g., the `&` in `fn(&T)`).  Such regions need not be
            // constrained by `encl_region` as they are placeholders
            // for regions that are as-yet-unknown.
            ret;
          }
          _ {}
        }

        alt fcx.mk_subr(encl_region, region) {
          result::err(_) {
            tcx.sess.span_err(
                span,
                #fmt["reference is not valid outside \
                      of its lifetime, %s",
                     ppaux::region_to_str(tcx, region)]);
          }
          result::ok(()) {
          }
        }
    }
}
