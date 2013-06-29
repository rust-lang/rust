// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #[warn(deprecated_mode)];

use core::prelude::*;

use middle::ty;

use middle::typeck::isr_alist;
use util::common::indenter;
use util::ppaux::region_to_str;
use util::ppaux;

use extra::list::Cons;

// Helper functions related to manipulating region types.

pub fn replace_bound_regions_in_fn_sig(
    tcx: ty::ctxt,
    isr: isr_alist,
    opt_self_ty: Option<ty::t>,
    fn_sig: &ty::FnSig,
    mapf: &fn(ty::bound_region) -> ty::Region)
    -> (isr_alist, Option<ty::t>, ty::FnSig)
{
    let mut all_tys = ty::tys_in_fn_sig(fn_sig);

    for opt_self_ty.iter().advance |&self_ty| {
        all_tys.push(self_ty);
    }

    for opt_self_ty.iter().advance |&t| { all_tys.push(t) }

    debug!("replace_bound_regions_in_fn_sig(self_ty=%?, fn_sig=%s, \
            all_tys=%?)",
           opt_self_ty.map(|&t| ppaux::ty_to_str(tcx, t)),
           ppaux::fn_sig_to_str(tcx, fn_sig),
           all_tys.map(|&t| ppaux::ty_to_str(tcx, t)));
    let _i = indenter();

    let isr = do create_bound_region_mapping(tcx, isr, all_tys) |br| {
        debug!("br=%?", br);
        mapf(br)
    };
    let new_fn_sig = ty::fold_sig(fn_sig, |t| {
        replace_bound_regions(tcx, isr, t)
    });
    let new_self_ty = opt_self_ty.map(|&t| replace_bound_regions(tcx, isr, t));

    debug!("result of replace_bound_regions_in_fn_sig: \
            new_self_ty=%?, \
            fn_sig=%s",
           new_self_ty.map(|&t| ppaux::ty_to_str(tcx, t)),
           ppaux::fn_sig_to_str(tcx, &new_fn_sig));

    return (isr, new_self_ty, new_fn_sig);

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
        to_r: &fn(ty::bound_region) -> ty::Region) -> isr_alist {

        // Takes `isr` (described above), `to_r` (described above),
        // and `r`, a region.  If `r` is anything other than a bound
        // region, or if it's a bound region that already appears in
        // `isr`, then we return `isr` unchanged.  If `r` is a bound
        // region that doesn't already appear in `isr`, we return an
        // updated isr_alist that now contains a mapping from `r` to
        // the result of calling `to_r` on it.
        fn append_isr(isr: isr_alist,
                      to_r: &fn(ty::bound_region) -> ty::Region,
                      r: ty::Region) -> isr_alist {
            match r {
              ty::re_empty | ty::re_free(*) | ty::re_static | ty::re_scope(_) |
              ty::re_infer(_) => {
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
        do tys.iter().fold(isr) |isr, ty| {
            let mut isr = isr;

            // Using fold_regions is inefficient, because it
            // constructs new types, but it avoids code duplication in
            // terms of locating all the regions within the various
            // kinds of types.  This had already caused me several
            // bugs so I decided to switch over.
            do ty::fold_regions(tcx, *ty) |r, in_fn| {
                if !in_fn { isr = append_isr(isr, |br| to_r(br), r); }
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
            let r1 = match r {
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
                             region_to_str(tcx, "", false, r)));
                  }
                }
              }

              // Free regions like these just stay the same:
              ty::re_empty |
              ty::re_static |
              ty::re_scope(_) |
              ty::re_free(*) |
              ty::re_infer(_) => r
            };
            r1
        }
    }
}

pub fn relate_nested_regions(
    tcx: ty::ctxt,
    opt_region: Option<ty::Region>,
    ty: ty::t,
    relate_op: &fn(ty::Region, ty::Region))
{
    /*!
     *
     * This rather specialized function walks each region `r` that appear
     * in `ty` and invokes `relate_op(r_encl, r)` for each one.  `r_encl`
     * here is the region of any enclosing `&'r T` pointer.  If there is
     * no enclosing pointer, and `opt_region` is Some, then `opt_region.get()`
     * is used instead.  Otherwise, no callback occurs at all).
     *
     * Here are some examples to give you an intution:
     *
     * - `relate_nested_regions(Some('r1), &'r2 uint)` invokes
     *   - `relate_op('r1, 'r2)`
     * - `relate_nested_regions(Some('r1), &'r2 &'r3 uint)` invokes
     *   - `relate_op('r1, 'r2)`
     *   - `relate_op('r2, 'r3)`
     * - `relate_nested_regions(None, &'r2 &'r3 uint)` invokes
     *   - `relate_op('r2, 'r3)`
     * - `relate_nested_regions(None, &'r2 &'r3 &'r4 uint)` invokes
     *   - `relate_op('r2, 'r3)`
     *   - `relate_op('r2, 'r4)`
     *   - `relate_op('r3, 'r4)`
     *
     * This function is used in various pieces of code because we enforce the
     * constraint that a region pointer cannot outlive the things it points at.
     * Hence, in the second example above, `'r2` must be a subregion of `'r3`.
     */

    let mut the_stack = ~[];
    for opt_region.iter().advance |&r| { the_stack.push(r); }
    walk_ty(tcx, &mut the_stack, ty, relate_op);

    fn walk_ty(tcx: ty::ctxt,
               the_stack: &mut ~[ty::Region],
               ty: ty::t,
               relate_op: &fn(ty::Region, ty::Region))
    {
        match ty::get(ty).sty {
            ty::ty_rptr(r, ref mt) |
            ty::ty_evec(ref mt, ty::vstore_slice(r)) => {
                relate(*the_stack, r, |x,y| relate_op(x,y));
                the_stack.push(r);
                walk_ty(tcx, the_stack, mt.ty, |x,y| relate_op(x,y));
                the_stack.pop();
            }
            _ => {
                ty::fold_regions_and_ty(
                    tcx,
                    ty,
                    |r| { relate(     *the_stack, r, |x,y| relate_op(x,y)); r },
                    |t| { walk_ty(tcx, the_stack, t, |x,y| relate_op(x,y)); t },
                    |t| { walk_ty(tcx, the_stack, t, |x,y| relate_op(x,y)); t });
            }
        }
    }

    fn relate(the_stack: &[ty::Region],
              r_sub: ty::Region,
              relate_op: &fn(ty::Region, ty::Region))
    {
        for the_stack.iter().advance |&r| {
            if !r.is_bound() && !r_sub.is_bound() {
                relate_op(r, r_sub);
            }
        }
    }
}

pub fn relate_free_regions(
    tcx: ty::ctxt,
    self_ty: Option<ty::t>,
    fn_sig: &ty::FnSig)
{
    /*!
     * This function populates the region map's `free_region_map`.
     * It walks over the transformed self type and argument types
     * for each function just before we check the body of that
     * function, looking for types where you have a borrowed
     * pointer to other borrowed data (e.g., `&'a &'b [uint]`.
     * We do not allow borrowed pointers to outlive the things they
     * point at, so we can assume that `'a <= 'b`.
     *
     * Tests: `src/test/compile-fail/regions-free-region-ordering-*.rs`
     */

    debug!("relate_free_regions >>");

    let mut all_tys = ~[];
    for fn_sig.inputs.iter().advance |arg| {
        all_tys.push(*arg);
    }
    for self_ty.iter().advance |&t| {
        all_tys.push(t);
    }

    for all_tys.iter().advance |&t| {
        debug!("relate_free_regions(t=%s)", ppaux::ty_to_str(tcx, t));
        relate_nested_regions(tcx, None, t, |a, b| {
            match (&a, &b) {
                (&ty::re_free(free_a), &ty::re_free(free_b)) => {
                    tcx.region_maps.relate_free_regions(free_a, free_b);
                }
                _ => {}
            }
        })
    }

    debug!("<< relate_free_regions");
}
