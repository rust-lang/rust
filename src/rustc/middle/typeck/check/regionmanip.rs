// #[warn(deprecated_mode)];
// #[warn(deprecated_pattern)];

use syntax::print::pprust::{expr_to_str};

// Helper functions related to manipulating region types.

fn replace_bound_regions_in_fn_ty(
    tcx: ty::ctxt,
    isr: isr_alist,
    self_info: Option<self_info>,
    fn_ty: &ty::fn_ty,
    mapf: fn(ty::bound_region) -> ty::region) ->
    {isr: isr_alist, self_info: Option<self_info>, fn_ty: ty::fn_ty} {

    // Take self_info apart; the self_ty part is the only one we want
    // to update here.
    let (self_ty, rebuild_self_info) = match self_info {
      Some(s) => (Some(s.self_ty), |t| Some({self_ty: t,.. s})),
      None => (None, |_t| None)
    };

    let mut all_tys = ty::tys_in_fn_ty(fn_ty);

    match self_info {
      Some({explicit_self: {node: ast::sty_region(m), _}, _}) => {
        let region = ty::re_bound(ty::br_self);
        let ty = ty::mk_rptr(tcx, region,
                             { ty: ty::mk_self(tcx), mutbl: m });
        vec::push(all_tys, ty);
      }
      _ => {}
    }


    for self_ty.each |t| { vec::push(all_tys, t) }

    debug!("replace_bound_regions_in_fn_ty(self_info.self_ty=%?, fn_ty=%s, \
                all_tys=%?)",
           self_ty.map(|t| ty_to_str(tcx, t)),
           ty_to_str(tcx, ty::mk_fn(tcx, *fn_ty)),
           all_tys.map(|t| ty_to_str(tcx, t)));
    let _i = indenter();

    let isr = do create_bound_region_mapping(tcx, isr, all_tys) |br| {
        debug!("br=%?", br);
        mapf(br)
    };
    let ty_fn = ty::ty_fn(*fn_ty);
    let t_fn = ty::fold_sty_to_ty(tcx, &ty_fn, |t| {
        replace_bound_regions(tcx, isr, t)
    });
    let t_self = self_ty.map(|t| replace_bound_regions(tcx, isr, t));

    debug!("result of replace_bound_regions_in_fn_ty: self_info.self_ty=%?, \
                fn_ty=%s",
           t_self.map(|t| ty_to_str(tcx, t)),
           ty_to_str(tcx, t_fn));


    // Glue updated self_ty back together with its original def_id.
    let new_self_info: Option<self_info> = match t_self {
      None    => None,
      Some(t) => rebuild_self_info(t)
    };

    return {isr: isr,
         self_info: new_self_info,
         fn_ty: match ty::get(t_fn).struct { ty::ty_fn(o) => o,
          _ => tcx.sess.bug(~"replace_bound_regions_in_fn_ty: impossible")}};


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
            match r {
              ty::re_free(_, _) | ty::re_static | ty::re_scope(_) |
              ty::re_var(_) => {
                isr
              }
              ty::re_bound(br) => {
                match isr.find(br) {
                  Some(_) => isr,
                  None => @Cons((br, to_r(br)), isr)
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
            match r {
              // As long as we are not within a fn() type, `&T` is
              // mapped to the free region anon_r.  But within a fn
              // type, it remains bound.
              ty::re_bound(ty::br_anon(_)) if in_fn => r,

              ty::re_bound(br) => {
                match isr.find(br) {
                  // In most cases, all named, bound regions will be
                  // mapped to some free region.
                  Some(fr) => fr,

                  // But in the case of a fn() type, there may be
                  // named regions within that remain bound:
                  None if in_fn => r,
                  None => {
                    tcx.sess.bug(
                        fmt!("Bound region not found in \
                              in_scope_regions list: %s",
                             region_to_str(tcx, r)));
                  }
                }
              }

              // Free regions like these just stay the same:
              ty::re_static |
              ty::re_scope(_) |
              ty::re_free(_, _) |
              ty::re_var(_) => r
            }
        }
    }
}
