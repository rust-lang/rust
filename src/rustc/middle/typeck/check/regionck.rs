/*

The region check is a final pass that runs over the AST after we have
inferred the type constraints but before we have actually finalized
the types.  Its purpose is to embed some final region constraints.
The reason that this is not done earlier is that sometimes we don't
know whether a given type will be a region pointer or not until this
phase.

In particular, we ensure that, if the type of an expression or
variable is `&r.T`, then the expression or variable must occur within
the region scope `r`.

*/

import util::ppaux;
import syntax::print::pprust;

type rcx = @{fcx: @fn_ctxt, mut errors_reported: uint};
type rvt = visit::vt<rcx>;

fn regionck_expr(fcx: @fn_ctxt, e: @ast::expr) {
    let rcx = @{fcx:fcx, mut errors_reported: 0u};
    let v = regionck_visitor();
    v.visit_expr(e, rcx, v);
}

fn regionck_fn(fcx: @fn_ctxt,
               _decl: ast::fn_decl,
               blk: ast::blk) {
    let rcx = @{fcx:fcx, mut errors_reported: 0u};
    let v = regionck_visitor();
    v.visit_block(blk, rcx, v);
}

fn regionck_visitor() -> rvt {
    visit::mk_vt(@{visit_item: visit_item,
                   visit_stmt: visit_stmt,
                   visit_expr: visit_expr,
                   visit_block: visit_block,
                   visit_pat: visit_pat,
                   visit_local: visit_local
                   with *visit::default_visitor()})
}

fn visit_item(_item: @ast::item, &&_rcx: rcx, _v: rvt) {
    // Ignore items
}

fn visit_local(l: @ast::local, &&rcx: rcx, v: rvt) {
    let e = rcx.errors_reported;
    v.visit_pat(l.node.pat, rcx, v);
    if e != rcx.errors_reported {
        ret; // if decl has errors, skip initializer expr
    }

    v.visit_ty(l.node.ty, rcx, v);
    for l.node.init.each { |i|
        v.visit_expr(i.expr, rcx, v);
    }
}

fn visit_pat(p: @ast::pat, &&rcx: rcx, v: rvt) {
    let fcx = rcx.fcx;
    alt p.node {
      ast::pat_ident(path, _)
      if !pat_util::pat_is_variant(fcx.ccx.tcx.def_map, p) {
        #debug["visit_pat binding=%s", *path.idents[0]];
        visit_node(p.id, p.span, rcx);
      }
      _ {}
    }

    visit::visit_pat(p, rcx, v);
}

fn visit_block(b: ast::blk, &&rcx: rcx, v: rvt) {
    visit::visit_block(b, rcx, v);
}

fn visit_expr(e: @ast::expr, &&rcx: rcx, v: rvt) {
    #debug["visit_expr(e=%s)", pprust::expr_to_str(e)];

    alt e.node {
      ast::expr_path(*) {
        // Avoid checking the use of local variables, as we already
        // check their definitions.  The def'n always encloses the
        // use.  So if the def'n is enclosed by the region, then the
        // uses will also be enclosed (and otherwise, an error will
        // have been reported at the def'n site).
        alt lookup_def(rcx.fcx, e.span, e.id) {
          ast::def_local(*) | ast::def_arg(*) | ast::def_upvar(*) { ret; }
          _ { }
        }
      }
      _ { }
    }

    if !visit_node(e.id, e.span, rcx) { ret; }
    visit::visit_expr(e, rcx, v);
}

fn visit_stmt(s: @ast::stmt, &&rcx: rcx, v: rvt) {
    visit::visit_stmt(s, rcx, v);
}

// checks the type of the node `id` and reports an error if it
// references a region that is not in scope for that node.  Returns
// false if an error is reported; this is used to cause us to cut off
// region checking for that subtree to avoid reporting tons of errors.
fn visit_node(id: ast::node_id, span: span, rcx: rcx) -> bool {
    let fcx = rcx.fcx;

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty0 = fcx.node_ty(id);
    let ty = alt infer::resolve_deep(fcx.infcx, ty0, false) {
      result::err(_) { ret true; }
      result::ok(ty) { ty }
    };

    // find the region where this expr evaluation is taking place
    let tcx = fcx.ccx.tcx;
    let encl_region = ty::encl_region(tcx, id);

    #debug["visit_node(ty=%s, id=%d, encl_region=%s, ty0=%s)",
           ppaux::ty_to_str(tcx, ty),
           id,
           ppaux::region_to_str(tcx, encl_region),
           ppaux::ty_to_str(tcx, ty0)];

    // Otherwise, look at the type and see if it is a region pointer.
    let e = rcx.errors_reported;
    ty::walk_regions_and_ty(
        tcx, ty,
        { |r| constrain_region(rcx, encl_region, span, r); },
        { |t| ty::type_has_regions(t) });
    ret (e == rcx.errors_reported);

    fn constrain_region(rcx: rcx,
                        encl_region: ty::region,
                        span: span,
                        region: ty::region) {
        let tcx = rcx.fcx.ccx.tcx;

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

        alt rcx.fcx.mk_subr(encl_region, region) {
          result::err(_) {
            tcx.sess.span_err(
                span,
                #fmt["reference is not valid outside \
                      of its lifetime, %s",
                     ppaux::region_to_str(tcx, region)]);
            rcx.errors_reported += 1u;
          }
          result::ok(()) {
          }
        }
    }
}
