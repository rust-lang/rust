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

pub use self::WfConstraint::*;

use middle::subst::{ParamSpace, Subst, Substs};
use middle::ty::{self, Ty};
use middle::ty_fold::{TypeFolder};

use syntax::ast;

use util::ppaux::Repr;

// Helper functions related to manipulating region types.

pub enum WfConstraint<'tcx> {
    RegionSubRegionConstraint(Option<Ty<'tcx>>, ty::Region, ty::Region),
    RegionSubParamConstraint(Option<Ty<'tcx>>, ty::Region, ty::ParamTy),
}

struct Wf<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    stack: Vec<(ty::Region, Option<Ty<'tcx>>)>,
    out: Vec<WfConstraint<'tcx>>,
}

/// This routine computes the well-formedness constraints that must hold for the type `ty` to
/// appear in a context with lifetime `outer_region`
pub fn region_wf_constraints<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    ty: Ty<'tcx>,
    outer_region: ty::Region)
    -> Vec<WfConstraint<'tcx>>
{
    let mut stack = Vec::new();
    stack.push((outer_region, None));
    let mut wf = Wf { tcx: tcx,
                      stack: stack,
                      out: Vec::new() };
    wf.accumulate_from_ty(ty);
    wf.out
}

impl<'a, 'tcx> Wf<'a, 'tcx> {
    fn accumulate_from_ty(&mut self, ty: Ty<'tcx>) {
        debug!("Wf::accumulate_from_ty(ty={})",
               ty.repr(self.tcx));

        match ty.sty {
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

            ty::ty_unboxed_closure(_, region, _) => {
                // An "unboxed closure type" is basically
                // modeled here as equivalent to a struct like
                //
                //     struct TheClosure<'b> {
                //         ...
                //     }
                //
                // where the `'b` is the lifetime bound of the
                // contents (i.e., all contents must outlive 'b).
                //
                // Even though unboxed closures are glorified structs
                // of upvars, we do not need to consider them as they
                // can't generate any new constraints.  The
                // substitutions on the closure are equal to the free
                // substitutions of the enclosing parameter
                // environment.  An upvar captured by value has the
                // same type as the original local variable which is
                // already checked for consistency.  If the upvar is
                // captured by reference it must also outlive the
                // region bound on the closure, but this is explicitly
                // handled by logic in regionck.
                self.push_region_constraint_from_top(*region);
            }

            ty::ty_trait(ref t) => {
                let required_region_bounds =
                    ty::object_region_bounds(self.tcx, Some(&t.principal), t.bounds.builtin_bounds);
                self.accumulate_from_object_ty(ty, t.bounds.region_bound, required_region_bounds)
            }

            ty::ty_enum(def_id, substs) |
            ty::ty_struct(def_id, substs) => {
                let item_scheme = ty::lookup_item_type(self.tcx, def_id);
                self.accumulate_from_adt(ty, def_id, &item_scheme.generics, substs)
            }

            ty::ty_vec(t, _) |
            ty::ty_ptr(ty::mt { ty: t, .. }) |
            ty::ty_uniq(t) => {
                self.accumulate_from_ty(t)
            }

            ty::ty_rptr(r_b, mt) => {
                self.accumulate_from_rptr(ty, *r_b, mt.ty);
            }

            ty::ty_param(p) => {
                self.push_param_constraint_from_top(p);
            }

            ty::ty_projection(ref data) => {
                // `<T as TraitRef<..>>::Name`

                // FIXME(#20303) -- gain ability to require that ty_projection : in-scope region,
                // like a type parameter

                // this seems like a minimal requirement:
                let trait_def = ty::lookup_trait_def(self.tcx, data.trait_ref.def_id);
                self.accumulate_from_adt(ty, data.trait_ref.def_id,
                                         &trait_def.generics, data.trait_ref.substs)
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
                            ty.repr(self.tcx))[]);
            }
        }
    }

    fn accumulate_from_rptr(&mut self,
                            ty: Ty<'tcx>,
                            r_b: ty::Region,
                            ty_b: Ty<'tcx>) {
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

    /// Pushes a constraint that `r_b` must outlive the top region on the stack.
    fn push_region_constraint_from_top(&mut self,
                                       r_b: ty::Region) {

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

    /// Pushes a constraint that `r_a <= r_b`, due to `opt_ty`
    fn push_sub_region_constraint(&mut self,
                                  opt_ty: Option<Ty<'tcx>>,
                                  r_a: ty::Region,
                                  r_b: ty::Region) {
        self.out.push(RegionSubRegionConstraint(opt_ty, r_a, r_b));
    }

    /// Pushes a constraint that `param_ty` must outlive the top region on the stack.
    fn push_param_constraint_from_top(&mut self,
                                      param_ty: ty::ParamTy) {
        let &(region, opt_ty) = self.stack.last().unwrap();
        self.push_param_constraint(region, opt_ty, param_ty);
    }

    /// Pushes a constraint that `region <= param_ty`, due to `opt_ty`
    fn push_param_constraint(&mut self,
                             region: ty::Region,
                             opt_ty: Option<Ty<'tcx>>,
                             param_ty: ty::ParamTy) {
        self.out.push(RegionSubParamConstraint(opt_ty, region, param_ty));
    }

    fn accumulate_from_adt(&mut self,
                           ty: Ty<'tcx>,
                           def_id: ast::DefId,
                           generics: &ty::Generics<'tcx>,
                           substs: &Substs<'tcx>)
    {
        // The generic declarations from the type, appropriately
        // substituted for the actual substitutions.
        let generics = generics.subst(self.tcx, substs);

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
                                  ty: Ty<'tcx>,
                                  c: &ty::ClosureTy<'tcx>)
    {
        match c.store {
            ty::RegionTraitStore(r_b, _) => {
                self.push_region_constraint_from_top(r_b);
            }
            ty::UniqTraitStore => { }
        }

        let required_region_bounds =
            ty::object_region_bounds(self.tcx, None, c.bounds.builtin_bounds);
        self.accumulate_from_object_ty(ty, c.bounds.region_bound, required_region_bounds);
    }

    fn accumulate_from_object_ty(&mut self,
                                 ty: Ty<'tcx>,
                                 region_bound: ty::Region,
                                 required_region_bounds: Vec<ty::Region>)
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
        let r_c = region_bound;
        self.push_region_constraint_from_top(r_c);

        // And then, in turn, to be well-formed, the
        // `region_bound` that user specified must imply the
        // region bounds required from all of the trait types:
        for &r_d in required_region_bounds.iter() {
            // Each of these is an instance of the `'c <= 'b`
            // constraint above
            self.out.push(RegionSubRegionConstraint(Some(ty), r_d, r_c));
        }
    }
}

impl<'tcx> Repr<'tcx> for WfConstraint<'tcx> {
    fn repr(&self, tcx: &ty::ctxt) -> String {
        match *self {
            RegionSubRegionConstraint(_, r_a, r_b) => {
                format!("RegionSubRegionConstraint({}, {})",
                        r_a.repr(tcx),
                        r_b.repr(tcx))
            }

            RegionSubParamConstraint(_, r, p) => {
                format!("RegionSubParamConstraint({}, {})",
                        r.repr(tcx),
                        p.repr(tcx))
            }
        }
    }
}
