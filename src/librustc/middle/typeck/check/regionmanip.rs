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

use middle::ty;
use middle::ty_fold;
use middle::ty_fold::TypeFolder;
use std::hashmap::HashMap;
use util::ppaux::Repr;
use util::ppaux;

// Helper functions related to manipulating region types.

pub fn replace_bound_regions_in_fn_sig(
        tcx: ty::ctxt,
        opt_self_ty: Option<ty::t>,
        fn_sig: &ty::FnSig,
        mapf: |ty::BoundRegion| -> ty::Region)
        -> (HashMap<ty::BoundRegion,ty::Region>, Option<ty::t>, ty::FnSig) {
    debug!("replace_bound_regions_in_fn_sig(self_ty={}, fn_sig={})",
            opt_self_ty.repr(tcx),
            fn_sig.repr(tcx));

    let mut map = HashMap::new();
    let (fn_sig, opt_self_ty) = {
        let mut f = ty_fold::RegionFolder::regions(tcx, |r| {
                debug!("region r={}", r.to_str());
                match r {
                ty::ReLateBound(s, br) if s == fn_sig.binder_id => {
                    *map.find_or_insert_with(br, |_| mapf(br))
                }
                _ => r
            }});
        (ty_fold::super_fold_sig(&mut f, fn_sig),
         ty_fold::fold_opt_ty(&mut f, opt_self_ty))
    };
    debug!("resulting map: {}", map.to_str());
    (map, opt_self_ty, fn_sig)
}

pub fn relate_nested_regions(tcx: ty::ctxt,
                             opt_region: Option<ty::Region>,
                             ty: ty::t,
                             relate_op: |ty::Region, ty::Region|) {
    /*!
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

    let mut rr = RegionRelator { tcx: tcx,
                                 stack: ~[],
                                 relate_op: relate_op };
    match opt_region {
        Some(o_r) => { rr.stack.push(o_r); }
        None => {}
    }
    rr.fold_ty(ty);

    struct RegionRelator<'a> {
        tcx: ty::ctxt,
        stack: ~[ty::Region],
        relate_op: 'a |ty::Region, ty::Region|,
    }

    // FIXME(#10151) -- Define more precisely when a region is
    // considered "nested". Consider taking variance into account as
    // well.

    impl<'a> TypeFolder for RegionRelator<'a> {
        fn tcx(&self) -> ty::ctxt {
            self.tcx
        }

        fn fold_ty(&mut self, ty: ty::t) -> ty::t {
            match ty::get(ty).sty {
                ty::ty_rptr(r, ref mt) |
                ty::ty_evec(ref mt, ty::vstore_slice(r)) => {
                    self.relate(r);
                    self.stack.push(r);
                    ty_fold::super_fold_ty(self, mt.ty);
                    self.stack.pop();
                }

                _ => {
                    ty_fold::super_fold_ty(self, ty);
                }
            }

            ty
        }

        fn fold_region(&mut self, r: ty::Region) -> ty::Region {
            self.relate(r);
            r
        }
    }

    impl<'a> RegionRelator<'a> {
        fn relate(&mut self, r_sub: ty::Region) {
            for &r in self.stack.iter() {
                if !r.is_bound() && !r_sub.is_bound() {
                    (self.relate_op)(r, r_sub);
                }
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
     * We do not allow references to outlive the things they
     * point at, so we can assume that `'a <= 'b`.
     *
     * Tests: `src/test/compile-fail/regions-free-region-ordering-*.rs`
     */

    debug!("relate_free_regions >>");

    let mut all_tys = ~[];
    for arg in fn_sig.inputs.iter() {
        all_tys.push(*arg);
    }
    for &t in self_ty.iter() {
        all_tys.push(t);
    }

    for &t in all_tys.iter() {
        debug!("relate_free_regions(t={})", ppaux::ty_to_str(tcx, t));
        relate_nested_regions(tcx, None, t, |a, b| {
            match (&a, &b) {
                (&ty::ReFree(free_a), &ty::ReFree(free_b)) => {
                    tcx.region_maps.relate_free_regions(free_a, free_b);
                }
                _ => {}
            }
        })
    }

    debug!("<< relate_free_regions");
}
