// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #![warn(deprecated_mode)]

use middle::subst::{ParamSpace, Subst, Substs};
use middle::ty;
use middle::ty_fold;
use middle::ty_fold::TypeFolder;

use syntax::ast;

use std::collections::HashMap;
use std::collections::hashmap::{Occupied, Vacant};
use util::ppaux::Repr;

// Helper functions related to manipulating region types.

pub fn replace_late_bound_regions_in_fn_sig(
        tcx: &ty::ctxt,
        fn_sig: &ty::FnSig,
        mapf: |ty::BoundRegion| -> ty::Region)
        -> (HashMap<ty::BoundRegion,ty::Region>, ty::FnSig) {
    debug!("replace_late_bound_regions_in_fn_sig({})", fn_sig.repr(tcx));

    let mut map = HashMap::new();
    let fn_sig = {
        let mut f = ty_fold::RegionFolder::regions(tcx, |r| {
            debug!("region r={}", r.to_string());
            match r {
                ty::ReLateBound(s, br) if s == fn_sig.binder_id => {
                    * match map.entry(br) {
                        Vacant(entry) => entry.set(mapf(br)),
                        Occupied(entry) => entry.into_mut(),
                    }
                }
                _ => r
            }
        });
        ty_fold::super_fold_sig(&mut f, fn_sig)
    };
    debug!("resulting map: {}", map);
    (map, fn_sig)
}

pub enum WfConstraint {
    RegionSubRegionConstraint(Option<ty::t>, ty::Region, ty::Region),
    RegionSubParamConstraint(Option<ty::t>, ty::Region, ty::ParamTy),
}

struct Wf<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    stack: Vec<(ty::Region, Option<ty::t>)>,
    out: Vec<WfConstraint>,
}

pub fn region_wf_constraints(
    tcx: &ty::ctxt,
    ty: ty::t,
    outer_region: ty::Region)
    -> Vec<WfConstraint>
{
    /*!
     * This routine computes the well-formedness constraints that must
     * hold for the type `ty` to appear in a context with lifetime
     * `outer_region`
     */

    let mut stack = Vec::new();
    stack.push((outer_region, None));
    let mut wf = Wf { tcx: tcx,
                      stack: stack,
                      out: Vec::new() };
    wf.accumulate_from_ty(ty);
    wf.out
}

impl<'a, 'tcx> Wf<'a, 'tcx> {
    fn accumulate_from_ty(&mut self, ty: ty::t) {
        debug!("Wf::accumulate_from_ty(ty={})",
               ty.repr(self.tcx));

        match ty::get(ty).sty {
            ty::ty_nil |
            ty::ty_bot |
            ty::ty_bool |
            ty::ty_char |
            ty::ty_int(..) |
            ty::ty_uint(..) |
            ty::ty_float(..) |
            ty::ty_bare_fn(..) |
            ty::ty_err |
            ty::ty_str => {
                // No borrowed content reachable here.
            }

            ty::ty_closure(box ref c) => {
                self.accumulate_from_closure_ty(ty, c);
            }

            ty::ty_unboxed_closure(_, region) => {
                // An "unboxed closure type" is basically
                // modeled here as equivalent to a struct like
                //
                //     struct TheClosure<'b> {
                //         ...
                //     }
                //
                // where the `'b` is the lifetime bound of the
                // contents (i.e., all contents must outlive 'b).
                self.push_region_constraint_from_top(region);
            }

            ty::ty_trait(ref t) => {
                self.accumulate_from_object_ty(ty, &t.bounds)
            }

            ty::ty_enum(def_id, ref substs) |
            ty::ty_struct(def_id, ref substs) => {
                self.accumulate_from_adt(ty, def_id, substs)
            }

            ty::ty_vec(t, _) |
            ty::ty_ptr(ty::mt { ty: t, .. }) |
            ty::ty_uniq(t) => {
                self.accumulate_from_ty(t)
            }

            ty::ty_rptr(r_b, mt) => {
                self.accumulate_from_rptr(ty, r_b, mt.ty);
            }

            ty::ty_param(p) => {
                self.push_param_constraint_from_top(p);
            }

            ty::ty_tup(ref tuptys) => {
                for &tupty in tuptys.iter() {
                    self.accumulate_from_ty(tupty);
                }
            }

            ty::ty_infer(_) => {
                // This should not happen, BUT:
                //
                //   Currently we uncover region relationships on
                //   entering the fn check. We should do this after
                //   the fn check, then we can call this case a bug().
            }

            ty::ty_open(_) => {
                self.tcx.sess.bug(
                    format!("Unexpected type encountered while doing wf check: {}",
                            ty.repr(self.tcx)).as_slice());
            }
        }
    }

    fn accumulate_from_rptr(&mut self,
                            ty: ty::t,
                            r_b: ty::Region,
                            ty_b: ty::t) {
        // We are walking down a type like this, and current
        // position is indicated by caret:
        //
        //     &'a &'b ty_b
        //         ^
        //
        // At this point, top of stack will be `'a`. We must
        // require that `'a <= 'b`.

        self.push_region_constraint_from_top(r_b);

        // Now we push `'b` onto the stack, because it must
        // constrain any borrowed content we find within `T`.

        self.stack.push((r_b, Some(ty)));
        self.accumulate_from_ty(ty_b);
        self.stack.pop().unwrap();
    }

    fn push_region_constraint_from_top(&mut self,
                                       r_b: ty::Region) {
        /*!
         * Pushes a constraint that `r_b` must outlive the
         * top region on the stack.
         */

        // Indicates that we have found borrowed content with a lifetime
        // of at least `r_b`. This adds a constraint that `r_b` must
        // outlive the region `r_a` on top of the stack.
        //
        // As an example, imagine walking a type like:
        //
        //     &'a &'b T
        //         ^
        //
        // when we hit the inner pointer (indicated by caret), `'a` will
        // be on top of stack and `'b` will be the lifetime of the content
        // we just found. So we add constraint that `'a <= 'b`.

        let &(r_a, opt_ty) = self.stack.last().unwrap();
        self.push_sub_region_constraint(opt_ty, r_a, r_b);
    }

    fn push_sub_region_constraint(&mut self,
                                  opt_ty: Option<ty::t>,
                                  r_a: ty::Region,
                                  r_b: ty::Region) {
        /*! Pushes a constraint that `r_a <= r_b`, due to `opt_ty` */
        self.out.push(RegionSubRegionConstraint(opt_ty, r_a, r_b));
    }

    fn push_param_constraint_from_top(&mut self,
                                      param_ty: ty::ParamTy) {
        /*!
         * Pushes a constraint that `param_ty` must outlive the
         * top region on the stack.
         */

        let &(region, opt_ty) = self.stack.last().unwrap();
        self.push_param_constraint(region, opt_ty, param_ty);
    }

    fn push_param_constraint(&mut self,
                             region: ty::Region,
                             opt_ty: Option<ty::t>,
                             param_ty: ty::ParamTy) {
        /*! Pushes a constraint that `region <= param_ty`, due to `opt_ty` */
        self.out.push(RegionSubParamConstraint(opt_ty, region, param_ty));
    }

    fn accumulate_from_adt(&mut self,
                           ty: ty::t,
                           def_id: ast::DefId,
                           substs: &Substs)
    {
        // The generic declarations from the type, appropriately
        // substituted for the actual substitutions.
        let generics =
            ty::lookup_item_type(self.tcx, def_id)
            .generics
            .subst(self.tcx, substs);

        // Variance of each type/region parameter.
        let variances = ty::item_variances(self.tcx, def_id);

        for &space in ParamSpace::all().iter() {
            let region_params = substs.regions().get_slice(space);
            let region_variances = variances.regions.get_slice(space);
            let region_param_defs = generics.regions.get_slice(space);
            assert_eq!(region_params.len(), region_variances.len());
            for (&region_param, (&region_variance, region_param_def)) in
                region_params.iter().zip(
                    region_variances.iter().zip(
                        region_param_defs.iter()))
            {
                match region_variance {
                    ty::Covariant | ty::Bivariant => {
                        // Ignore covariant or bivariant region
                        // parameters.  To understand why, consider a
                        // struct `Foo<'a>`. If `Foo` contains any
                        // references with lifetime `'a`, then `'a` must
                        // be at least contravariant (and possibly
                        // invariant). The only way to have a covariant
                        // result is if `Foo` contains only a field with a
                        // type like `fn() -> &'a T`; i.e., a bare
                        // function that can produce a reference of
                        // lifetime `'a`. In this case, there is no
                        // *actual data* with lifetime `'a` that is
                        // reachable. (Presumably this bare function is
                        // really returning static data.)
                    }

                    ty::Contravariant | ty::Invariant => {
                        // If the parameter is contravariant or
                        // invariant, there may indeed be reachable
                        // data with this lifetime. See other case for
                        // more details.
                        self.push_region_constraint_from_top(region_param);
                    }
                }

                for &region_bound in region_param_def.bounds.iter() {
                    // The type declared a constraint like
                    //
                    //     'b : 'a
                    //
                    // which means that `'a <= 'b` (after
                    // substitution).  So take the region we
                    // substituted for `'a` (`region_bound`) and make
                    // it a subregion of the region we substituted
                    // `'b` (`region_param`).
                    self.push_sub_region_constraint(
                        Some(ty), region_bound, region_param);
                }
            }

            let types = substs.types.get_slice(space);
            let type_variances = variances.types.get_slice(space);
            let type_param_defs = generics.types.get_slice(space);
            assert_eq!(types.len(), type_variances.len());
            for (&type_param_ty, (&variance, type_param_def)) in
                types.iter().zip(
                    type_variances.iter().zip(
                        type_param_defs.iter()))
            {
                debug!("type_param_ty={} variance={}",
                       type_param_ty.repr(self.tcx),
                       variance.repr(self.tcx));

                match variance {
                    ty::Contravariant | ty::Bivariant => {
                        // As above, except that in this it is a
                        // *contravariant* reference that indices that no
                        // actual data of type T is reachable.
                    }

                    ty::Covariant | ty::Invariant => {
                        self.accumulate_from_ty(type_param_ty);
                    }
                }

                // Inspect bounds on this type parameter for any
                // region bounds.
                for &r in type_param_def.bounds.region_bounds.iter() {
                    self.stack.push((r, Some(ty)));
                    self.accumulate_from_ty(type_param_ty);
                    self.stack.pop().unwrap();
                }
            }
        }
    }

    fn accumulate_from_closure_ty(&mut self,
                                  ty: ty::t,
                                  c: &ty::ClosureTy)
    {
        match c.store {
            ty::RegionTraitStore(r_b, _) => {
                self.push_region_constraint_from_top(r_b);
            }
            ty::UniqTraitStore => { }
        }

        self.accumulate_from_object_ty(ty, &c.bounds)
    }

    fn accumulate_from_object_ty(&mut self,
                                 ty: ty::t,
                                 bounds: &ty::ExistentialBounds)
    {
        // Imagine a type like this:
        //
        //     trait Foo { }
        //     trait Bar<'c> : 'c { }
        //
        //     &'b (Foo+'c+Bar<'d>)
        //         ^
        //
        // In this case, the following relationships must hold:
        //
        //     'b <= 'c
        //     'd <= 'c
        //
        // The first conditions is due to the normal region pointer
        // rules, which say that a reference cannot outlive its
        // referent.
        //
        // The final condition may be a bit surprising. In particular,
        // you may expect that it would have been `'c <= 'd`, since
        // usually lifetimes of outer things are conservative
        // approximations for inner things. However, it works somewhat
        // differently with trait objects: here the idea is that if the
        // user specifies a region bound (`'c`, in this case) it is the
        // "master bound" that *implies* that bounds from other traits are
        // all met. (Remember that *all bounds* in a type like
        // `Foo+Bar+Zed` must be met, not just one, hence if we write
        // `Foo<'x>+Bar<'y>`, we know that the type outlives *both* 'x and
        // 'y.)
        //
        // Note: in fact we only permit builtin traits, not `Bar<'d>`, I
        // am looking forward to the future here.

        // The content of this object type must outlive
        // `bounds.region_bound`:
        let r_c = bounds.region_bound;
        self.push_region_constraint_from_top(r_c);

        // And then, in turn, to be well-formed, the
        // `region_bound` that user specified must imply the
        // region bounds required from all of the trait types:
        let required_region_bounds =
            ty::required_region_bounds(self.tcx,
                                       [],
                                       bounds.builtin_bounds,
                                       []);
        for &r_d in required_region_bounds.iter() {
            // Each of these is an instance of the `'c <= 'b`
            // constraint above
            self.out.push(RegionSubRegionConstraint(Some(ty), r_d, r_c));
        }
    }
}
