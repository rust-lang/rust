/*
 * The region checking pass. Ensures that region-annotated pointers never
 * outlive their referents.
 */

import driver::session::session;
import middle::ty;
import std::map::hashmap;
import syntax::{ast, visit};
import util::ppaux;

fn check_expr(expr: @ast::expr,
              &&tcx: ty::ctxt,
              visitor: visit::vt<ty::ctxt>) {
    visit::visit_expr(expr, tcx, visitor);

    let t = ty::expr_ty(tcx, expr);
    if !ty::type_has_regions(t) { ret; }
    ty::walk_ty(t) { |t|
        alt ty::get(t).struct {
          ty::ty_rptr(region, _) {
            alt region {
              ty::re_bound(_) | ty::re_free(_, _) | ty::re_static |
              ty::re_var(_) {
                /* ok */
              }
              ty::re_scope(id) {
                if !region::scope_contains(tcx.region_map, id, expr.id) {
                    tcx.sess.span_err(
                        expr.span,
                        #fmt["reference is not valid outside of %s",
                             ppaux::re_scope_id_to_str(tcx, id)]);
                }
              }
            }
          }
          _ { /* no-op */ }
        }
    }
}

fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {
    let visitor = visit::mk_vt(@{
        visit_expr: check_expr
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, tcx, visitor);
}

