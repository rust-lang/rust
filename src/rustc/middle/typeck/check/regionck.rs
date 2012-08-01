/*

The region check is a final pass that runs over the AST after we have
inferred the type constraints but before we have actually finalized
the types.  Its purpose is to embed some final region constraints.
The reason that this is not done earlier is that sometimes we don't
know whether a given type will be a region pointer or not until this
phase.

In particular, we ensure that, if the type of an expression or
variable is `&r/T`, then the expression or variable must occur within
the region scope `r`.  Note that in some cases `r` may still be a
region variable, so this gives us a chance to influence the value for
`r` that we infer to ensure we choose a value large enough to enclose
all uses.  There is a lengthy comment in visit_node() that explains
this point a bit better.

*/

import util::ppaux;
import syntax::print::pprust;
import infer::{resolve_type, resolve_all, force_all,
               resolve_rvar, force_rvar, fres};
import middle::kind::check_owned;

enum rcx { rcx_({fcx: @fn_ctxt, mut errors_reported: uint}) }
type rvt = visit::vt<@rcx>;

impl methods for @rcx {
    /// Try to resolve the type for the given node.
    ///
    /// Note one important point: we do not attempt to resolve *region
    /// variables* here.  This is because regionck is essentially adding
    /// constraints to those region variables and so may yet influence
    /// how they are resolved.
    ///
    /// Consider this silly example:
    ///
    ///     fn borrow(x: &int) -> &int {x}
    ///     fn foo(x: @int) -> int {  /* block: B */
    ///         let b = borrow(x);    /* region: <R0> */
    ///         *b
    ///     }
    ///
    /// Here, the region of `b` will be `<R0>`.  `<R0>` is constrainted
    /// to be some subregion of the block B and some superregion of
    /// the call.  If we forced it now, we'd choose the smaller region
    /// (the call).  But that would make the *b illegal.  Since we don't
    /// resolve, the type of b will be `&<R0>.int` and then `*b` will require
    /// that `<R0>` be bigger than the let and the `*b` expression, so we
    /// will effectively resolve `<R0>` to be the block B.
    fn resolve_type(unresolved_ty: ty::t) -> fres<ty::t> {
        resolve_type(self.fcx.infcx, unresolved_ty,
                     (resolve_all | force_all) -
                     (resolve_rvar | force_rvar))
    }

    /// Try to resolve the type for the given node.
    fn resolve_node_type(id: ast::node_id) -> fres<ty::t> {
        self.resolve_type(self.fcx.node_ty(id))
    }
}

fn regionck_expr(fcx: @fn_ctxt, e: @ast::expr) {
    let rcx = rcx_({fcx:fcx, mut errors_reported: 0u});
    let v = regionck_visitor();
    v.visit_expr(e, @rcx, v);
}

fn regionck_fn(fcx: @fn_ctxt,
               _decl: ast::fn_decl,
               blk: ast::blk) {
    let rcx = rcx_({fcx:fcx, mut errors_reported: 0u});
    let v = regionck_visitor();
    v.visit_block(blk, @rcx, v);
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

fn visit_item(_item: @ast::item, &&_rcx: @rcx, _v: rvt) {
    // Ignore items
}

fn visit_local(l: @ast::local, &&rcx: @rcx, v: rvt) {
    let e = rcx.errors_reported;
    v.visit_pat(l.node.pat, rcx, v);
    if e != rcx.errors_reported {
        ret; // if decl has errors, skip initializer expr
    }

    v.visit_ty(l.node.ty, rcx, v);
    for l.node.init.each |i| {
        v.visit_expr(i.expr, rcx, v);
    }
}

fn visit_pat(p: @ast::pat, &&rcx: @rcx, v: rvt) {
    let fcx = rcx.fcx;
    alt p.node {
      ast::pat_ident(_, path, _)
      if !pat_util::pat_is_variant(fcx.ccx.tcx.def_map, p) {
        debug!{"visit_pat binding=%s", *path.idents[0]};
        visit_node(p.id, p.span, rcx);
      }
      _ {}
    }

    visit::visit_pat(p, rcx, v);
}

fn visit_block(b: ast::blk, &&rcx: @rcx, v: rvt) {
    visit::visit_block(b, rcx, v);
}

fn visit_expr(e: @ast::expr, &&rcx: @rcx, v: rvt) {
    debug!{"visit_expr(e=%s)", pprust::expr_to_str(e)};

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

      ast::expr_cast(source, _) {
        // Determine if we are casting `source` to an trait instance.
        // If so, we have to be sure that the type of the source obeys
        // the trait's region bound.
        //
        // Note: there is a subtle point here concerning type
        // parameters.  It is possible that the type of `source`
        // contains type parameters, which in turn may contain regions
        // that are not visible to us (only the caller knows about
        // them).  The kind checker is ultimately responsible for
        // guaranteeing region safety in that particular case.  There
        // is an extensive comment on the function
        // check_cast_for_escaping_regions() in kind.rs explaining how
        // it goes about doing that.
        alt rcx.resolve_node_type(e.id) {
          result::err(_) => { ret; /* typeck will fail anyhow */ }
          result::ok(target_ty) => {
            alt ty::get(target_ty).struct {
              ty::ty_trait(_, substs) {
                let trait_region = alt substs.self_r {
                  some(r) => {r}
                  none => {ty::re_static}
                };
                let source_ty = rcx.fcx.expr_ty(source);
                constrain_regions_in_type(rcx, trait_region,
                                          e.span, source_ty);
              }
              _ { }
            }
          }
        };

      }

      _ { }
    }

    if !visit_node(e.id, e.span, rcx) { ret; }
    visit::visit_expr(e, rcx, v);
}

fn visit_stmt(s: @ast::stmt, &&rcx: @rcx, v: rvt) {
    visit::visit_stmt(s, rcx, v);
}

// checks the type of the node `id` and reports an error if it
// references a region that is not in scope for that node.  Returns
// false if an error is reported; this is used to cause us to cut off
// region checking for that subtree to avoid reporting tons of errors.
fn visit_node(id: ast::node_id, span: span, rcx: @rcx) -> bool {
    let fcx = rcx.fcx;

    // Try to resolve the type.  If we encounter an error, then typeck
    // is going to fail anyway, so just stop here and let typeck
    // report errors later on in the writeback phase.
    let ty = alt rcx.resolve_node_type(id) {
      result::err(_) { ret true; }
      result::ok(ty) { ty }
    };

    // find the region where this expr evaluation is taking place
    let tcx = fcx.ccx.tcx;
    let encl_region = ty::encl_region(tcx, id);

    debug!{"visit_node(ty=%s, id=%d, encl_region=%s)",
           ppaux::ty_to_str(tcx, ty),
           id,
           ppaux::region_to_str(tcx, encl_region)};

    // Otherwise, look at the type and see if it is a region pointer.
    ret constrain_regions_in_type(rcx, encl_region, span, ty);
}

fn constrain_regions_in_type(
    rcx: @rcx,
    encl_region: ty::region,
    span: span,
    ty: ty::t) -> bool {

    let e = rcx.errors_reported;
    ty::walk_regions_and_ty(
        rcx.fcx.ccx.tcx, ty,
        |r| constrain_region(rcx, encl_region, span, r),
        |t| ty::type_has_regions(t));
    ret (e == rcx.errors_reported);

    fn constrain_region(rcx: @rcx,
                        encl_region: ty::region,
                        span: span,
                        region: ty::region) {
        let tcx = rcx.fcx.ccx.tcx;

        debug!{"constrain_region(encl_region=%s, region=%s)",
               ppaux::region_to_str(tcx, encl_region),
               ppaux::region_to_str(tcx, region)};

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
            let region1 = rcx.fcx.infcx.resolve_region_if_possible(region);
            tcx.sess.span_err(
                span,
                fmt!{"reference is not valid outside \
                      of its lifetime, %s",
                     ppaux::region_to_str(tcx, region1)});
            rcx.errors_reported += 1u;
          }
          result::ok(()) {
          }
        }
    }
}
