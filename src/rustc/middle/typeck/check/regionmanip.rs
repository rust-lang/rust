import syntax::print::pprust::{expr_to_str};

// Helper functions related to manipulating region types.

// Extracts the bound regions from bound_tys and then replaces those same
// regions in `sty` with fresh region variables, returning the resulting type.
// Does not descend into fn types.  This is used when deciding whether an impl
// applies at a given call site.  See also universally_quantify_before_call().
fn universally_quantify_from_sty(fcx: @fn_ctxt,
                                 span: span,
                                 bound_tys: [ty::t],
                                 sty: ty::sty) -> ty::t {

    #debug["universally_quantify_from_sty(bound_tys=%?)",
           bound_tys.map {|x| fcx.ty_to_str(x) }];
    indent {||
        let tcx = fcx.tcx();
        let isr = collect_bound_regions_in_tys(tcx, @nil, bound_tys) { |br|
            let rvar = fcx.infcx.next_region_var();
            #debug["Bound region %s maps to %s",
                   bound_region_to_str(fcx.ccx.tcx, br),
                   region_to_str(fcx.ccx.tcx, rvar)];
            rvar
        };
        let t_res = ty::fold_sty_to_ty(fcx.ccx.tcx, sty) { |t|
            replace_bound_regions(tcx, span, isr, t)
        };
        #debug["Result of universal quant. is %s", fcx.ty_to_str(t_res)];
        t_res
    }
}

fn replace_bound_regions(
    tcx: ty::ctxt,
    span: span,
    isr: isr_alist,
    ty: ty::t) -> ty::t {

    ty::fold_regions(tcx, ty) { |r, in_fn|
        alt r {
          // As long as we are not within a fn() type, `&T` is mapped to the
          // free region anon_r.  But within a fn type, it remains bound.
          ty::re_bound(ty::br_anon) if in_fn { r }

          ty::re_bound(br) {
            alt isr.find(br) {
              // In most cases, all named, bound regions will be mapped to
              // some free region.
              some(fr) { fr }

              // But in the case of a fn() type, there may be named regions
              // within that remain bound:
              none if in_fn { r }
              none {
                tcx.sess.span_bug(
                    span,
                    #fmt["Bound region not found in \
                          in_scope_regions list: %s",
                         region_to_str(tcx, r)]);
              }
            }
          }

          // Free regions like these just stay the same:
          ty::re_static |
          ty::re_scope(_) |
          ty::re_free(_, _) |
          ty::re_var(_) { r }
        }
    }
}

/* Returns the region that &expr should be placed into.  If expr is an
 * lvalue, this will be the region of the lvalue.  Otherwise, if region is
 * an rvalue, the semantics are that the result is stored into a temporary
 * stack position and so the resulting region will be the enclosing block.
 */
fn region_of(fcx: @fn_ctxt, expr: @ast::expr) -> ty::region {
    #debug["region_of(expr=%s)", expr_to_str(expr)];
    ret alt expr.node {
      ast::expr_path(path) {
        def(fcx, expr, lookup_def(fcx, path.span, expr.id))}
      ast::expr_field(base, _, _) {
        deref(fcx, base)}
      ast::expr_index(base, _) {
        deref(fcx, base)}
      ast::expr_unary(ast::deref, base) {
        deref(fcx, base)}
      _ {
        borrow(fcx, expr)}
    };

    fn borrow(fcx: @fn_ctxt, expr: @ast::expr) -> ty::region {
        ty::encl_region(fcx.ccx.tcx, expr.id)
    }

    fn deref(fcx: @fn_ctxt, base: @ast::expr) -> ty::region {
        let base_ty = fcx.expr_ty(base);
        let base_ty = structurally_resolved_type(fcx, base.span, base_ty);
        alt ty::get(base_ty).struct {
          ty::ty_rptr(region, _) { region }
          ty::ty_box(_) | ty::ty_uniq(_) { borrow(fcx, base) }
          _ { region_of(fcx, base) }
        }
    }

    fn def(fcx: @fn_ctxt, expr: @ast::expr, d: ast::def) -> ty::region {
        alt d {
          ast::def_arg(local_id, _) |
          ast::def_local(local_id, _) |
          ast::def_binding(local_id) {
            #debug["region_of.def/arg/local/binding(id=%d)", local_id];
            let local_scope = fcx.ccx.tcx.region_map.get(local_id);
            ty::re_scope(local_scope)
          }
          ast::def_upvar(_, inner, _) {
            #debug["region_of.def/upvar"];
            def(fcx, expr, *inner)
          }
          ast::def_self(*) {
            alt fcx.in_scope_regions.find(ty::br_self) {
              some(r) {r}
              none {
                // eventually, this should never happen... self should
                // always be an &self.T rptr
                borrow(fcx, expr)
              }
            }
          }
          ast::def_fn(_, _) | ast::def_mod(_) |
          ast::def_native_mod(_) | ast::def_const(_) |
          ast::def_use(_) | ast::def_variant(_, _) |
          ast::def_ty(_) | ast::def_prim_ty(_) |
          ast::def_ty_param(_, _) | ast::def_class(_) |
          ast::def_region(_) {
            ty::re_static
          }
        }
    }
}

fn collect_bound_regions_in_tys(
    tcx: ty::ctxt,
    isr: isr_alist,
    tys: [ty::t],
    to_r: fn(ty::bound_region) -> ty::region) -> isr_alist {

    tys.foldl(isr) { |isr, t|
        collect_bound_regions_in_ty(tcx, isr, t, to_r)
    }
}

fn collect_bound_regions_in_ty(
    tcx: ty::ctxt,
    isr: isr_alist,
    ty: ty::t,
    to_r: fn(ty::bound_region) -> ty::region) -> isr_alist {

    fn append_isr(isr: isr_alist,
                  to_r: fn(ty::bound_region) -> ty::region,
                  r: ty::region) -> isr_alist {
        alt r {
          ty::re_free(_, _) | ty::re_static | ty::re_scope(_) |
          ty::re_var(_) {
            isr
          }
          ty::re_bound(br) {
            alt isr.find(br) {
              some(_) { isr }
              none { @cons((br, to_r(br)), isr) }
            }
          }
        }
    }

    let mut isr = isr;

    // Using fold_regions is inefficient, because it constructs new types, but
    // it avoids code duplication in terms of locating all the regions within
    // the various kinds of types.  This had already caused me several bugs
    // so I decided to switch over.
    ty::fold_regions(tcx, ty) { |r, in_fn|
        if !in_fn { isr = append_isr(isr, to_r, r); }
        r
    };

    ret isr;
}
