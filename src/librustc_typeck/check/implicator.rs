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

use astconv::object_region_bounds;
use middle::infer::{InferCtxt, GenericKind};
use middle::subst::{Substs};
use middle::traits;
use middle::ty::{self, ToPolyTraitRef, Ty};
use middle::ty_fold::{TypeFoldable, TypeFolder};

use std::rc::Rc;
use syntax::ast;
use syntax::codemap::Span;

use util::common::ErrorReported;
use util::ppaux::Repr;

// Helper functions related to manipulating region types.

pub enum Implication<'tcx> {
    RegionSubRegion(Option<Ty<'tcx>>, ty::Region, ty::Region),
    RegionSubGeneric(Option<Ty<'tcx>>, ty::Region, GenericKind<'tcx>),
    Predicate(ast::DefId, ty::Predicate<'tcx>),
}

struct Implicator<'a, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a,'tcx>,
    closure_typer: &'a (ty::ClosureTyper<'tcx>+'a),
    body_id: ast::NodeId,
    stack: Vec<(ty::Region, Option<Ty<'tcx>>)>,
    span: Span,
    out: Vec<Implication<'tcx>>,
}

/// This routine computes the well-formedness constraints that must hold for the type `ty` to
/// appear in a context with lifetime `outer_region`
pub fn implications<'a,'tcx>(
    infcx: &'a InferCtxt<'a,'tcx>,
    closure_typer: &ty::ClosureTyper<'tcx>,
    body_id: ast::NodeId,
    ty: Ty<'tcx>,
    outer_region: ty::Region,
    span: Span)
    -> Vec<Implication<'tcx>>
{
    debug!("implications(body_id={}, ty={}, outer_region={})",
           body_id,
           ty.repr(closure_typer.tcx()),
           outer_region.repr(closure_typer.tcx()));

    let mut stack = Vec::new();
    stack.push((outer_region, None));
    let mut wf = Implicator { closure_typer: closure_typer,
                              infcx: infcx,
                              body_id: body_id,
                              span: span,
                              stack: stack,
                              out: Vec::new() };
    wf.accumulate_from_ty(ty);
    debug!("implications: out={}", wf.out.repr(closure_typer.tcx()));
    wf.out
}

impl<'a, 'tcx> Implicator<'a, 'tcx> {
    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn accumulate_from_ty(&mut self, ty: Ty<'tcx>) {
        debug!("accumulate_from_ty(ty={})",
               ty.repr(self.tcx()));

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

            ty::ty_closure(_, region, _) => {
                // An "closure type" is basically
                // modeled here as equivalent to a struct like
                //
                //     struct TheClosure<'b> {
                //         ...
                //     }
                //
                // where the `'b` is the lifetime bound of the
                // contents (i.e., all contents must outlive 'b).
                //
                // Even though closures are glorified structs
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
                    object_region_bounds(self.tcx(), &t.principal, t.bounds.builtin_bounds);
                self.accumulate_from_object_ty(ty, t.bounds.region_bound, required_region_bounds)
            }

            ty::ty_enum(def_id, substs) |
            ty::ty_struct(def_id, substs) => {
                let item_scheme = ty::lookup_item_type(self.tcx(), def_id);
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

                self.push_projection_constraint_from_top(data);
            }

            ty::ty_tup(ref tuptys) => {
                for &tupty in tuptys {
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
                self.tcx().sess.bug(
                    &format!("Unexpected type encountered while doing wf check: {}",
                            ty.repr(self.tcx()))[]);
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
        self.out.push(Implication::RegionSubRegion(opt_ty, r_a, r_b));
    }

    /// Pushes a constraint that `param_ty` must outlive the top region on the stack.
    fn push_param_constraint_from_top(&mut self,
                                      param_ty: ty::ParamTy) {
        let &(region, opt_ty) = self.stack.last().unwrap();
        self.push_param_constraint(region, opt_ty, param_ty);
    }

    /// Pushes a constraint that `projection_ty` must outlive the top region on the stack.
    fn push_projection_constraint_from_top(&mut self,
                                           projection_ty: &ty::ProjectionTy<'tcx>) {
        let &(region, opt_ty) = self.stack.last().unwrap();
        self.out.push(Implication::RegionSubGeneric(
            opt_ty, region, GenericKind::Projection(projection_ty.clone())));
    }

    /// Pushes a constraint that `region <= param_ty`, due to `opt_ty`
    fn push_param_constraint(&mut self,
                             region: ty::Region,
                             opt_ty: Option<Ty<'tcx>>,
                             param_ty: ty::ParamTy) {
        self.out.push(Implication::RegionSubGeneric(
            opt_ty, region, GenericKind::Param(param_ty)));
    }

    fn accumulate_from_adt(&mut self,
                           ty: Ty<'tcx>,
                           def_id: ast::DefId,
                           _generics: &ty::Generics<'tcx>,
                           substs: &Substs<'tcx>)
    {
        let predicates =
            ty::lookup_predicates(self.tcx(), def_id).instantiate(self.tcx(), substs);
        let predicates = match self.fully_normalize(&predicates) {
            Ok(predicates) => predicates,
            Err(ErrorReported) => { return; }
        };

        for predicate in predicates.predicates.as_slice() {
            match *predicate {
                ty::Predicate::Trait(ref data) => {
                    self.accumulate_from_assoc_types_transitive(data);
                }
                ty::Predicate::Equate(..) => { }
                ty::Predicate::Projection(..) => { }
                ty::Predicate::RegionOutlives(ref data) => {
                    match ty::no_late_bound_regions(self.tcx(), data) {
                        None => { }
                        Some(ty::OutlivesPredicate(r_a, r_b)) => {
                            self.push_sub_region_constraint(Some(ty), r_b, r_a);
                        }
                    }
                }
                ty::Predicate::TypeOutlives(ref data) => {
                    match ty::no_late_bound_regions(self.tcx(), data) {
                        None => { }
                        Some(ty::OutlivesPredicate(ty_a, r_b)) => {
                            self.stack.push((r_b, Some(ty)));
                            self.accumulate_from_ty(ty_a);
                            self.stack.pop().unwrap();
                        }
                    }
                }
            }
        }

        let obligations = predicates.predicates
                                    .into_iter()
                                    .map(|pred| Implication::Predicate(def_id, pred));
        self.out.extend(obligations);

        let variances = ty::item_variances(self.tcx(), def_id);

        for (&region, &variance) in substs.regions().iter().zip(variances.regions.iter()) {
            match variance {
                ty::Contravariant | ty::Invariant => {
                    // If any data with this lifetime is reachable
                    // within, it must be at least contravariant.
                    self.push_region_constraint_from_top(region)
                }
                ty::Covariant | ty::Bivariant => { }
            }
        }

        for (&ty, &variance) in substs.types.iter().zip(variances.types.iter()) {
            match variance {
                ty::Covariant | ty::Invariant => {
                    // If any data of this type is reachable within,
                    // it must be at least covariant.
                    self.accumulate_from_ty(ty);
                }
                ty::Contravariant | ty::Bivariant => { }
            }
        }
    }

    /// Given that there is a requirement that `Foo<X> : 'a`, where
    /// `Foo` is declared like `struct Foo<T> where T : SomeTrait`,
    /// this code finds all the associated types defined in
    /// `SomeTrait` (and supertraits) and adds a requirement that `<X
    /// as SomeTrait>::N : 'a` (where `N` is some associated type
    /// defined in `SomeTrait`). This rule only applies to
    /// trait-bounds that are not higher-ranked, because we cannot
    /// project out of a HRTB. This rule helps code using associated
    /// types to compile, see Issue #22246 for an example.
    fn accumulate_from_assoc_types_transitive(&mut self,
                                              data: &ty::PolyTraitPredicate<'tcx>)
    {
        debug!("accumulate_from_assoc_types_transitive({})",
               data.repr(self.tcx()));

        for poly_trait_ref in traits::supertraits(self.tcx(), data.to_poly_trait_ref()) {
            match ty::no_late_bound_regions(self.tcx(), &poly_trait_ref) {
                Some(trait_ref) => { self.accumulate_from_assoc_types(trait_ref); }
                None => { }
            }
        }
    }

    fn accumulate_from_assoc_types(&mut self,
                                   trait_ref: Rc<ty::TraitRef<'tcx>>)
    {
        debug!("accumulate_from_assoc_types({})",
               trait_ref.repr(self.tcx()));

        let trait_def_id = trait_ref.def_id;
        let trait_def = ty::lookup_trait_def(self.tcx(), trait_def_id);
        let assoc_type_projections: Vec<_> =
            trait_def.associated_type_names
                     .iter()
                     .map(|&name| ty::mk_projection(self.tcx(), trait_ref.clone(), name))
                     .collect();
        debug!("accumulate_from_assoc_types: assoc_type_projections={}",
               assoc_type_projections.repr(self.tcx()));
        let tys = match self.fully_normalize(&assoc_type_projections) {
            Ok(tys) => { tys }
            Err(ErrorReported) => { return; }
        };
        for ty in tys {
            self.accumulate_from_ty(ty);
        }
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
        for &r_d in &required_region_bounds {
            // Each of these is an instance of the `'c <= 'b`
            // constraint above
            self.out.push(Implication::RegionSubRegion(Some(ty), r_d, r_c));
        }
    }

    fn fully_normalize<T>(&self, value: &T) -> Result<T,ErrorReported>
        where T : TypeFoldable<'tcx> + ty::HasProjectionTypes + Clone + Repr<'tcx>
    {
        let value =
            traits::fully_normalize(self.infcx,
                                    self.closure_typer,
                                    traits::ObligationCause::misc(self.span, self.body_id),
                                    value);
        match value {
            Ok(value) => Ok(value),
            Err(errors) => {
                // I don't like reporting these errors here, but I
                // don't know where else to report them just now. And
                // I don't really expect errors to arise here
                // frequently. I guess the best option would be to
                // propagate them out.
                traits::report_fulfillment_errors(self.infcx, &errors);
                Err(ErrorReported)
            }
        }
    }
}

impl<'tcx> Repr<'tcx> for Implication<'tcx> {
    fn repr(&self, tcx: &ty::ctxt<'tcx>) -> String {
        match *self {
            Implication::RegionSubRegion(_, ref r_a, ref r_b) => {
                format!("RegionSubRegion({}, {})",
                        r_a.repr(tcx),
                        r_b.repr(tcx))
            }

            Implication::RegionSubGeneric(_, ref r, ref p) => {
                format!("RegionSubGeneric({}, {})",
                        r.repr(tcx),
                        p.repr(tcx))
            }

            Implication::Predicate(ref def_id, ref p) => {
                format!("Predicate({}, {})",
                        def_id.repr(tcx),
                        p.repr(tcx))
            }
        }
    }
}
