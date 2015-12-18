// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::DefId;
use middle::infer::InferCtxt;
use middle::ty::outlives::{self, Component};
use middle::subst::Substs;
use middle::traits;
use middle::ty::{self, ToPredicate, Ty, TypeFoldable};
use std::iter::once;
use syntax::ast;
use syntax::codemap::Span;
use util::common::ErrorReported;

/// Returns the set of obligations needed to make `ty` well-formed.
/// If `ty` contains unresolved inference variables, this may include
/// further WF obligations. However, if `ty` IS an unresolved
/// inference variable, returns `None`, because we are not able to
/// make any progress at all. This is to prevent "livelock" where we
/// say "$0 is WF if $0 is WF".
pub fn obligations<'a,'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                            body_id: ast::NodeId,
                            ty: Ty<'tcx>,
                            span: Span)
                            -> Option<Vec<traits::PredicateObligation<'tcx>>>
{
    let mut wf = WfPredicates { infcx: infcx,
                                body_id: body_id,
                                span: span,
                                out: vec![] };
    if wf.compute(ty) {
        debug!("wf::obligations({:?}, body_id={:?}) = {:?}", ty, body_id, wf.out);
        let result = wf.normalize();
        debug!("wf::obligations({:?}, body_id={:?}) ~~> {:?}", ty, body_id, result);
        Some(result)
    } else {
        None // no progress made, return None
    }
}

/// Returns the obligations that make this trait reference
/// well-formed.  For example, if there is a trait `Set` defined like
/// `trait Set<K:Eq>`, then the trait reference `Foo: Set<Bar>` is WF
/// if `Bar: Eq`.
pub fn trait_obligations<'a,'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                  body_id: ast::NodeId,
                                  trait_ref: &ty::TraitRef<'tcx>,
                                  span: Span)
                                  -> Vec<traits::PredicateObligation<'tcx>>
{
    let mut wf = WfPredicates { infcx: infcx, body_id: body_id, span: span, out: vec![] };
    wf.compute_trait_ref(trait_ref);
    wf.normalize()
}

pub fn predicate_obligations<'a,'tcx>(infcx: &InferCtxt<'a, 'tcx>,
                                      body_id: ast::NodeId,
                                      predicate: &ty::Predicate<'tcx>,
                                      span: Span)
                                      -> Vec<traits::PredicateObligation<'tcx>>
{
    let mut wf = WfPredicates { infcx: infcx, body_id: body_id, span: span, out: vec![] };

    // (*) ok to skip binders, because wf code is prepared for it
    match *predicate {
        ty::Predicate::Trait(ref t) => {
            wf.compute_trait_ref(&t.skip_binder().trait_ref); // (*)
        }
        ty::Predicate::Equate(ref t) => {
            wf.compute(t.skip_binder().0);
            wf.compute(t.skip_binder().1);
        }
        ty::Predicate::RegionOutlives(..) => {
        }
        ty::Predicate::TypeOutlives(ref t) => {
            wf.compute(t.skip_binder().0);
        }
        ty::Predicate::Projection(ref t) => {
            let t = t.skip_binder(); // (*)
            wf.compute_projection(t.projection_ty);
            wf.compute(t.ty);
        }
        ty::Predicate::WellFormed(t) => {
            wf.compute(t);
        }
        ty::Predicate::ObjectSafe(_) => {
        }
    }

    wf.normalize()
}

/// Implied bounds are region relationships that we deduce
/// automatically.  The idea is that (e.g.) a caller must check that a
/// function's argument types are well-formed immediately before
/// calling that fn, and hence the *callee* can assume that its
/// argument types are well-formed. This may imply certain relationships
/// between generic parameters. For example:
///
///     fn foo<'a,T>(x: &'a T)
///
/// can only be called with a `'a` and `T` such that `&'a T` is WF.
/// For `&'a T` to be WF, `T: 'a` must hold. So we can assume `T: 'a`.
#[derive(Debug)]
pub enum ImpliedBound<'tcx> {
    RegionSubRegion(ty::Region, ty::Region),
    RegionSubParam(ty::Region, ty::ParamTy),
    RegionSubProjection(ty::Region, ty::ProjectionTy<'tcx>),
}

/// Compute the implied bounds that a callee/impl can assume based on
/// the fact that caller/projector has ensured that `ty` is WF.  See
/// the `ImpliedBound` type for more details.
pub fn implied_bounds<'a,'tcx>(
    infcx: &'a InferCtxt<'a,'tcx>,
    body_id: ast::NodeId,
    ty: Ty<'tcx>,
    span: Span)
    -> Vec<ImpliedBound<'tcx>>
{
    // Sometimes when we ask what it takes for T: WF, we get back that
    // U: WF is required; in that case, we push U onto this stack and
    // process it next. Currently (at least) these resulting
    // predicates are always guaranteed to be a subset of the original
    // type, so we need not fear non-termination.
    let mut wf_types = vec![ty];

    let mut implied_bounds = vec![];

    while let Some(ty) = wf_types.pop() {
        // Compute the obligations for `ty` to be well-formed. If `ty` is
        // an unresolved inference variable, just substituted an empty set
        // -- because the return type here is going to be things we *add*
        // to the environment, it's always ok for this set to be smaller
        // than the ultimate set. (Note: normally there won't be
        // unresolved inference variables here anyway, but there might be
        // during typeck under some circumstances.)
        let obligations = obligations(infcx, body_id, ty, span).unwrap_or(vec![]);

        // From the full set of obligations, just filter down to the
        // region relationships.
        implied_bounds.extend(
            obligations
            .into_iter()
            .flat_map(|obligation| {
                assert!(!obligation.has_escaping_regions());
                match obligation.predicate {
                    ty::Predicate::Trait(..) |
                    ty::Predicate::Equate(..) |
                    ty::Predicate::Projection(..) |
                    ty::Predicate::ObjectSafe(..) =>
                        vec![],

                    ty::Predicate::WellFormed(subty) => {
                        wf_types.push(subty);
                        vec![]
                    }

                    ty::Predicate::RegionOutlives(ref data) =>
                        match infcx.tcx.no_late_bound_regions(data) {
                            None =>
                                vec![],
                            Some(ty::OutlivesPredicate(r_a, r_b)) =>
                                vec![ImpliedBound::RegionSubRegion(r_b, r_a)],
                        },

                    ty::Predicate::TypeOutlives(ref data) =>
                        match infcx.tcx.no_late_bound_regions(data) {
                            None => vec![],
                            Some(ty::OutlivesPredicate(ty_a, r_b)) => {
                                let components = outlives::components(infcx, ty_a);
                                implied_bounds_from_components(r_b, components)
                            }
                        },
                }}));
    }

    implied_bounds
}

/// When we have an implied bound that `T: 'a`, we can further break
/// this down to determine what relationships would have to hold for
/// `T: 'a` to hold. We get to assume that the caller has validated
/// those relationships.
fn implied_bounds_from_components<'tcx>(sub_region: ty::Region,
                                        sup_components: Vec<Component<'tcx>>)
                                        -> Vec<ImpliedBound<'tcx>>
{
    sup_components
        .into_iter()
        .flat_map(|component| {
            match component {
                Component::Region(r) =>
                    vec!(ImpliedBound::RegionSubRegion(sub_region, r)),
                Component::Param(p) =>
                    vec!(ImpliedBound::RegionSubParam(sub_region, p)),
                Component::Projection(p) =>
                    vec!(ImpliedBound::RegionSubProjection(sub_region, p)),
                Component::EscapingProjection(_) =>
                    // If the projection has escaping regions, don't
                    // try to infer any implied bounds even for its
                    // free components. This is conservative, because
                    // the caller will still have to prove that those
                    // free components outlive `sub_region`. But the
                    // idea is that the WAY that the caller proves
                    // that may change in the future and we want to
                    // give ourselves room to get smarter here.
                    vec!(),
                Component::UnresolvedInferenceVariable(..) =>
                    vec!(),
            }
        })
        .collect()
}

struct WfPredicates<'a,'tcx:'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    body_id: ast::NodeId,
    span: Span,
    out: Vec<traits::PredicateObligation<'tcx>>,
}

impl<'a,'tcx> WfPredicates<'a,'tcx> {
    fn cause(&mut self, code: traits::ObligationCauseCode<'tcx>) -> traits::ObligationCause<'tcx> {
        traits::ObligationCause::new(self.span, self.body_id, code)
    }

    fn normalize(&mut self) -> Vec<traits::PredicateObligation<'tcx>> {
        let cause = self.cause(traits::MiscObligation);
        let infcx = &mut self.infcx;
        self.out.iter()
                .inspect(|pred| assert!(!pred.has_escaping_regions()))
                .flat_map(|pred| {
                    let mut selcx = traits::SelectionContext::new(infcx);
                    let pred = traits::normalize(&mut selcx, cause.clone(), pred);
                    once(pred.value).chain(pred.obligations)
                })
                .collect()
    }

    /// Pushes the obligations required for `trait_ref` to be WF into
    /// `self.out`.
    fn compute_trait_ref(&mut self, trait_ref: &ty::TraitRef<'tcx>) {
        let obligations = self.nominal_obligations(trait_ref.def_id, trait_ref.substs);
        self.out.extend(obligations);

        let cause = self.cause(traits::MiscObligation);
        self.out.extend(
            trait_ref.substs.types
                            .as_slice()
                            .iter()
                            .filter(|ty| !ty.has_escaping_regions())
                            .map(|ty| traits::Obligation::new(cause.clone(),
                                                              ty::Predicate::WellFormed(ty))));
    }

    /// Pushes the obligations required for `trait_ref::Item` to be WF
    /// into `self.out`.
    fn compute_projection(&mut self, data: ty::ProjectionTy<'tcx>) {
        // A projection is well-formed if (a) the trait ref itself is
        // WF WF and (b) the trait-ref holds.  (It may also be
        // normalizable and be WF that way.)

        self.compute_trait_ref(&data.trait_ref);

        if !data.has_escaping_regions() {
            let predicate = data.trait_ref.to_predicate();
            let cause = self.cause(traits::ProjectionWf(data));
            self.out.push(traits::Obligation::new(cause, predicate));
        }
    }

    /// Push new obligations into `out`. Returns true if it was able
    /// to generate all the predicates needed to validate that `ty0`
    /// is WF. Returns false if `ty0` is an unresolved type variable,
    /// in which case we are not able to simplify at all.
    fn compute(&mut self, ty0: Ty<'tcx>) -> bool {
        let mut subtys = ty0.walk();
        while let Some(ty) = subtys.next() {
            match ty.sty {
                ty::TyBool |
                ty::TyChar |
                ty::TyInt(..) |
                ty::TyUint(..) |
                ty::TyFloat(..) |
                ty::TyError |
                ty::TyStr |
                ty::TyParam(_) => {
                    // WfScalar, WfParameter, etc
                }

                ty::TySlice(subty) |
                ty::TyArray(subty, _) => {
                    if !subty.has_escaping_regions() {
                        let cause = self.cause(traits::SliceOrArrayElem);
                        match traits::trait_ref_for_builtin_bound(self.infcx.tcx,
                                                                  ty::BoundSized,
                                                                  subty) {
                            Ok(trait_ref) => {
                                self.out.push(
                                    traits::Obligation::new(cause,
                                                            trait_ref.to_predicate()));
                            }
                            Err(ErrorReported) => { }
                        }
                    }
                }

                ty::TyBox(_) |
                ty::TyTuple(_) |
                ty::TyRawPtr(_) => {
                    // simple cases that are WF if their type args are WF
                }

                ty::TyProjection(data) => {
                    subtys.skip_current_subtree(); // subtree handled by compute_projection
                    self.compute_projection(data);
                }

                ty::TyEnum(def, substs) |
                ty::TyStruct(def, substs) => {
                    // WfNominalType
                    let obligations = self.nominal_obligations(def.did, substs);
                    self.out.extend(obligations);
                }

                ty::TyRef(r, mt) => {
                    // WfReference
                    if !r.has_escaping_regions() && !mt.ty.has_escaping_regions() {
                        let cause = self.cause(traits::ReferenceOutlivesReferent(ty));
                        self.out.push(
                            traits::Obligation::new(
                                cause,
                                ty::Predicate::TypeOutlives(
                                    ty::Binder(
                                        ty::OutlivesPredicate(mt.ty, *r)))));
                    }
                }

                ty::TyClosure(..) => {
                    // the types in a closure are always the types of
                    // local variables (or possibly references to local
                    // variables), we'll walk those.
                    //
                    // (Though, local variables are probably not
                    // needed, as they are separately checked w/r/t
                    // WFedness.)
                }

                ty::TyBareFn(..) => {
                    // let the loop iterator into the argument/return
                    // types appearing in the fn signature
                }

                ty::TyTrait(ref data) => {
                    // WfObject
                    //
                    // Here, we defer WF checking due to higher-ranked
                    // regions. This is perhaps not ideal.
                    self.from_object_ty(ty, data);

                    // FIXME(#27579) RFC also considers adding trait
                    // obligations that don't refer to Self and
                    // checking those

                    let cause = self.cause(traits::MiscObligation);
                    self.out.push(
                        traits::Obligation::new(
                            cause,
                            ty::Predicate::ObjectSafe(data.principal_def_id())));
                }

                // Inference variables are the complicated case, since we don't
                // know what type they are. We do two things:
                //
                // 1. Check if they have been resolved, and if so proceed with
                //    THAT type.
                // 2. If not, check whether this is the type that we
                //    started with (ty0). In that case, we've made no
                //    progress at all, so return false. Otherwise,
                //    we've at least simplified things (i.e., we went
                //    from `Vec<$0>: WF` to `$0: WF`, so we can
                //    register a pending obligation and keep
                //    moving. (Goal is that an "inductive hypothesis"
                //    is satisfied to ensure termination.)
                ty::TyInfer(_) => {
                    let ty = self.infcx.shallow_resolve(ty);
                    if let ty::TyInfer(_) = ty.sty { // not yet resolved...
                        if ty == ty0 { // ...this is the type we started from! no progress.
                            return false;
                        }

                        let cause = self.cause(traits::MiscObligation);
                        self.out.push( // ...not the type we started from, so we made progress.
                            traits::Obligation::new(cause, ty::Predicate::WellFormed(ty)));
                    } else {
                        // Yes, resolved, proceed with the
                        // result. Should never return false because
                        // `ty` is not a TyInfer.
                        assert!(self.compute(ty));
                    }
                }
            }
        }

        // if we made it through that loop above, we made progress!
        return true;
    }

    fn nominal_obligations(&mut self,
                           def_id: DefId,
                           substs: &Substs<'tcx>)
                           -> Vec<traits::PredicateObligation<'tcx>>
    {
        let predicates =
            self.infcx.tcx.lookup_predicates(def_id)
                          .instantiate(self.infcx.tcx, substs);
        let cause = self.cause(traits::ItemObligation(def_id));
        predicates.predicates
                  .into_iter()
                  .map(|pred| traits::Obligation::new(cause.clone(), pred))
                  .filter(|pred| !pred.has_escaping_regions())
                  .collect()
    }

    fn from_object_ty(&mut self, ty: Ty<'tcx>, data: &ty::TraitTy<'tcx>) {
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

        if !data.has_escaping_regions() {
            let implicit_bounds =
                object_region_bounds(self.infcx.tcx,
                                     &data.principal,
                                     data.bounds.builtin_bounds);

            let explicit_bound = data.bounds.region_bound;

            for implicit_bound in implicit_bounds {
                let cause = self.cause(traits::ReferenceOutlivesReferent(ty));
                let outlives = ty::Binder(ty::OutlivesPredicate(explicit_bound, implicit_bound));
                self.out.push(traits::Obligation::new(cause, outlives.to_predicate()));
            }
        }
    }
}

/// Given an object type like `SomeTrait+Send`, computes the lifetime
/// bounds that must hold on the elided self type. These are derived
/// from the declarations of `SomeTrait`, `Send`, and friends -- if
/// they declare `trait SomeTrait : 'static`, for example, then
/// `'static` would appear in the list. The hard work is done by
/// `ty::required_region_bounds`, see that for more information.
pub fn object_region_bounds<'tcx>(
    tcx: &ty::ctxt<'tcx>,
    principal: &ty::PolyTraitRef<'tcx>,
    others: ty::BuiltinBounds)
    -> Vec<ty::Region>
{
    // Since we don't actually *know* the self type for an object,
    // this "open(err)" serves as a kind of dummy standin -- basically
    // a skolemized type.
    let open_ty = tcx.mk_infer(ty::FreshTy(0));

    // Note that we preserve the overall binding levels here.
    assert!(!open_ty.has_escaping_regions());
    let substs = tcx.mk_substs(principal.0.substs.with_self_ty(open_ty));
    let trait_refs = vec!(ty::Binder(ty::TraitRef::new(principal.0.def_id, substs)));

    let mut predicates = others.to_predicates(tcx, open_ty);
    predicates.extend(trait_refs.iter().map(|t| t.to_predicate()));

    tcx.required_region_bounds(open_ty, predicates)
}
