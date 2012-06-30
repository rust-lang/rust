import syntax::print::pprust::{expr_to_str};

// Helper functions related to manipulating region types.

fn replace_bound_regions_in_fn_ty(
    tcx: ty::ctxt,
    isr: isr_alist,
    self_ty: option<ty::t>,
    fn_ty: ty::fn_ty,
    mapf: fn(ty::bound_region) -> ty::region) -> {isr: isr_alist,
                                                  self_ty: option<ty::t>,
                                                  fn_ty: ty::fn_ty} {

    let mut all_tys = ty::tys_in_fn_ty(fn_ty);
    for self_ty.each |t| { vec::push(all_tys, t) }

    #debug["replace_bound_regions_in_fn_ty(self_ty=%?, fn_ty=%s, all_tys=%?)",
           self_ty.map(|t| ty_to_str(tcx, t)),
           ty_to_str(tcx, ty::mk_fn(tcx, fn_ty)),
           all_tys.map(|t| ty_to_str(tcx, t))];
    let _i = indenter();

    let isr = do create_bound_region_mapping(tcx, isr, all_tys) |br| {
        #debug["br=%?", br];
        mapf(br)
    };
    let t_fn = ty::fold_sty_to_ty(tcx, ty::ty_fn(fn_ty), |t| {
        replace_bound_regions(tcx, isr, t)
    });
    let t_self = self_ty.map(|t| replace_bound_regions(tcx, isr, t));

    #debug["result of replace_bound_regions_in_fn_ty: self_ty=%?, fn_ty=%s",
           t_self.map(|t| ty_to_str(tcx, t)),
           ty_to_str(tcx, t_fn)];

    ret {isr: isr,
         self_ty: t_self,
         fn_ty: alt check ty::get(t_fn).struct { ty::ty_fn(o) {o} }};


    // Takes `isr`, a (possibly empty) mapping from in-scope region
    // names ("isr"s) to their corresponding regions; `tys`, a list of
    // types, and `to_r`, a closure that takes a bound_region and
    // returns a region.  Returns an updated version of `isr`,
    // extended with the in-scope region names from all of the bound
    // regions appearing in the types in the `tys` list (if they're
    // not in `isr` already), with each of those in-scope region names
    // mapped to a region that's the result of applying `to_r` to
    // itself.
    fn create_bound_region_mapping(
        tcx: ty::ctxt,
        isr: isr_alist,
        tys: ~[ty::t],
        to_r: fn(ty::bound_region) -> ty::region) -> isr_alist {

        // Takes `isr` (described above), `to_r` (described above),
        // and `r`, a region.  If `r` is anything other than a bound
        // region, or if it's a bound region that already appears in
        // `isr`, then we return `isr` unchanged.  If `r` is a bound
        // region that doesn't already appear in `isr`, we return an
        // updated isr_alist that now contains a mapping from `r` to
        // the result of calling `to_r` on it.
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

        // For each type `ty` in `tys`...
        do tys.foldl(isr) |isr, ty| {
            let mut isr = isr;

            // Using fold_regions is inefficient, because it
            // constructs new types, but it avoids code duplication in
            // terms of locating all the regions within the various
            // kinds of types.  This had already caused me several
            // bugs so I decided to switch over.
            do ty::fold_regions(tcx, ty) |r, in_fn| {
                if !in_fn { isr = append_isr(isr, to_r, r); }
                r
            };

            isr
        }
    }

    // Takes `isr`, a mapping from in-scope region names ("isr"s) to
    // their corresponding regions; and `ty`, a type.  Returns an
    // updated version of `ty`, in which bound regions in `ty` have
    // been replaced with the corresponding bindings in `isr`.
    fn replace_bound_regions(
        tcx: ty::ctxt,
        isr: isr_alist,
        ty: ty::t) -> ty::t {

        do ty::fold_regions(tcx, ty) |r, in_fn| {
            alt r {
              // As long as we are not within a fn() type, `&T` is
              // mapped to the free region anon_r.  But within a fn
              // type, it remains bound.
              ty::re_bound(ty::br_anon) if in_fn { r }

              ty::re_bound(br) {
                alt isr.find(br) {
                  // In most cases, all named, bound regions will be
                  // mapped to some free region.
                  some(fr) { fr }

                  // But in the case of a fn() type, there may be
                  // named regions within that remain bound:
                  none if in_fn { r }
                  none {
                    tcx.sess.bug(
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
          ast::def_foreign_mod(_) | ast::def_const(_) |
          ast::def_use(_) | ast::def_variant(_, _) |
          ast::def_ty(_) | ast::def_prim_ty(_) |
          ast::def_ty_param(_, _) | ast::def_class(_) |
          ast::def_region(_) {
            ty::re_static
          }
        }
    }
}
