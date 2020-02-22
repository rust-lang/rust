use crate::infer::InferCtxt;
use crate::opaque_types::required_region_bounds;
use crate::traits::{self, AssocTypeBoundData};
use rustc::middle::lang_items;
use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_span::symbol::{kw, Ident};
use rustc_span::Span;

/// Returns the set of obligations needed to make `ty` well-formed.
/// If `ty` contains unresolved inference variables, this may include
/// further WF obligations. However, if `ty` IS an unresolved
/// inference variable, returns `None`, because we are not able to
/// make any progress at all. This is to prevent "livelock" where we
/// say "$0 is WF if $0 is WF".
pub fn obligations<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
    ty: Ty<'tcx>,
    span: Span,
) -> Option<Vec<traits::PredicateObligation<'tcx>>> {
    let mut wf = WfPredicates { infcx, param_env, body_id, span, out: vec![], item: None };
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
pub fn trait_obligations<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
    trait_ref: &ty::TraitRef<'tcx>,
    span: Span,
    item: Option<&'tcx hir::Item<'tcx>>,
) -> Vec<traits::PredicateObligation<'tcx>> {
    let mut wf = WfPredicates { infcx, param_env, body_id, span, out: vec![], item };
    wf.compute_trait_ref(trait_ref, Elaborate::All);
    wf.normalize()
}

pub fn predicate_obligations<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
    predicate: &ty::Predicate<'tcx>,
    span: Span,
) -> Vec<traits::PredicateObligation<'tcx>> {
    let mut wf = WfPredicates { infcx, param_env, body_id, span, out: vec![], item: None };

    // (*) ok to skip binders, because wf code is prepared for it
    match *predicate {
        ty::Predicate::Trait(ref t, _) => {
            wf.compute_trait_ref(&t.skip_binder().trait_ref, Elaborate::None); // (*)
        }
        ty::Predicate::RegionOutlives(..) => {}
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
        ty::Predicate::ObjectSafe(_) => {}
        ty::Predicate::ClosureKind(..) => {}
        ty::Predicate::Subtype(ref data) => {
            wf.compute(data.skip_binder().a); // (*)
            wf.compute(data.skip_binder().b); // (*)
        }
        ty::Predicate::ConstEvaluatable(def_id, substs) => {
            let obligations = wf.nominal_obligations(def_id, substs);
            wf.out.extend(obligations);

            for ty in substs.types() {
                wf.compute(ty);
            }
        }
    }

    wf.normalize()
}

struct WfPredicates<'a, 'tcx> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
    span: Span,
    out: Vec<traits::PredicateObligation<'tcx>>,
    item: Option<&'tcx hir::Item<'tcx>>,
}

/// Controls whether we "elaborate" supertraits and so forth on the WF
/// predicates. This is a kind of hack to address #43784. The
/// underlying problem in that issue was a trait structure like:
///
/// ```
/// trait Foo: Copy { }
/// trait Bar: Foo { }
/// impl<T: Bar> Foo for T { }
/// impl<T> Bar for T { }
/// ```
///
/// Here, in the `Foo` impl, we will check that `T: Copy` holds -- but
/// we decide that this is true because `T: Bar` is in the
/// where-clauses (and we can elaborate that to include `T:
/// Copy`). This wouldn't be a problem, except that when we check the
/// `Bar` impl, we decide that `T: Foo` must hold because of the `Foo`
/// impl. And so nowhere did we check that `T: Copy` holds!
///
/// To resolve this, we elaborate the WF requirements that must be
/// proven when checking impls. This means that (e.g.) the `impl Bar
/// for T` will be forced to prove not only that `T: Foo` but also `T:
/// Copy` (which it won't be able to do, because there is no `Copy`
/// impl for `T`).
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum Elaborate {
    All,
    None,
}

impl<'a, 'tcx> WfPredicates<'a, 'tcx> {
    fn cause(&mut self, code: traits::ObligationCauseCode<'tcx>) -> traits::ObligationCause<'tcx> {
        traits::ObligationCause::new(self.span, self.body_id, code)
    }

    fn normalize(&mut self) -> Vec<traits::PredicateObligation<'tcx>> {
        let cause = self.cause(traits::MiscObligation);
        let infcx = &mut self.infcx;
        let param_env = self.param_env;
        let mut obligations = Vec::with_capacity(self.out.len());
        for pred in &self.out {
            assert!(!pred.has_escaping_bound_vars());
            let mut selcx = traits::SelectionContext::new(infcx);
            let i = obligations.len();
            let value =
                traits::normalize_to(&mut selcx, param_env, cause.clone(), pred, &mut obligations);
            obligations.insert(i, value);
        }
        obligations
    }

    /// Pushes the obligations required for `trait_ref` to be WF into `self.out`.
    fn compute_trait_ref(&mut self, trait_ref: &ty::TraitRef<'tcx>, elaborate: Elaborate) {
        let tcx = self.infcx.tcx;
        let obligations = self.nominal_obligations(trait_ref.def_id, trait_ref.substs);

        let cause = self.cause(traits::MiscObligation);
        let param_env = self.param_env;

        let item = &self.item;
        let extend_cause_with_original_assoc_item_obligation =
            |cause: &mut traits::ObligationCause<'_>,
             pred: &ty::Predicate<'_>,
             trait_assoc_items: &[ty::AssocItem]| {
                let trait_item = tcx
                    .hir()
                    .as_local_hir_id(trait_ref.def_id)
                    .and_then(|trait_id| tcx.hir().find(trait_id));
                let (trait_name, trait_generics) = match trait_item {
                    Some(hir::Node::Item(hir::Item {
                        ident,
                        kind: hir::ItemKind::Trait(.., generics, _, _),
                        ..
                    }))
                    | Some(hir::Node::Item(hir::Item {
                        ident,
                        kind: hir::ItemKind::TraitAlias(generics, _),
                        ..
                    })) => (Some(ident), Some(generics)),
                    _ => (None, None),
                };

                let item_span = item.map(|i| tcx.sess.source_map().def_span(i.span));
                match pred {
                    ty::Predicate::Projection(proj) => {
                        // The obligation comes not from the current `impl` nor the `trait` being
                        // implemented, but rather from a "second order" obligation, like in
                        // `src/test/ui/associated-types/point-at-type-on-obligation-failure.rs`:
                        //
                        //   error[E0271]: type mismatch resolving `<Foo2 as Bar2>::Ok == ()`
                        //     --> $DIR/point-at-type-on-obligation-failure.rs:13:5
                        //      |
                        //   LL |     type Ok;
                        //      |          -- associated type defined here
                        //   ...
                        //   LL | impl Bar for Foo {
                        //      | ---------------- in this `impl` item
                        //   LL |     type Ok = ();
                        //      |     ^^^^^^^^^^^^^ expected `u32`, found `()`
                        //      |
                        //      = note: expected type `u32`
                        //                 found type `()`
                        //
                        // FIXME: we would want to point a span to all places that contributed to this
                        // obligation. In the case above, it should be closer to:
                        //
                        //   error[E0271]: type mismatch resolving `<Foo2 as Bar2>::Ok == ()`
                        //     --> $DIR/point-at-type-on-obligation-failure.rs:13:5
                        //      |
                        //   LL |     type Ok;
                        //      |          -- associated type defined here
                        //   LL |     type Sibling: Bar2<Ok=Self::Ok>;
                        //      |     -------------------------------- obligation set here
                        //   ...
                        //   LL | impl Bar for Foo {
                        //      | ---------------- in this `impl` item
                        //   LL |     type Ok = ();
                        //      |     ^^^^^^^^^^^^^ expected `u32`, found `()`
                        //   ...
                        //   LL | impl Bar2 for Foo2 {
                        //      | ---------------- in this `impl` item
                        //   LL |     type Ok = u32;
                        //      |     -------------- obligation set here
                        //      |
                        //      = note: expected type `u32`
                        //                 found type `()`
                        if let Some(hir::ItemKind::Impl { items, .. }) = item.map(|i| &i.kind) {
                            let trait_assoc_item = tcx.associated_item(proj.projection_def_id());
                            if let Some(impl_item) =
                                items.iter().find(|item| item.ident == trait_assoc_item.ident)
                            {
                                cause.span = impl_item.span;
                                cause.code = traits::AssocTypeBound(Box::new(AssocTypeBoundData {
                                    impl_span: item_span,
                                    original: trait_assoc_item.ident.span,
                                    bounds: vec![],
                                }));
                            }
                        }
                    }
                    ty::Predicate::Trait(proj, _) => {
                        // An associated item obligation born out of the `trait` failed to be met.
                        // Point at the `impl` that failed the obligation, the associated item that
                        // needed to meet the obligation, and the definition of that associated item,
                        // which should hold the obligation in most cases. An example can be seen in
                        // `src/test/ui/associated-types/point-at-type-on-obligation-failure-2.rs`:
                        //
                        //   error[E0277]: the trait bound `bool: Bar` is not satisfied
                        //     --> $DIR/point-at-type-on-obligation-failure-2.rs:8:5
                        //      |
                        //   LL |     type Assoc: Bar;
                        //      |          ----- associated type defined here
                        //   ...
                        //   LL | impl Foo for () {
                        //      | --------------- in this `impl` item
                        //   LL |     type Assoc = bool;
                        //      |     ^^^^^^^^^^^^^^^^^^ the trait `Bar` is not implemented for `bool`
                        //
                        // If the obligation comes from the where clause in the `trait`, we point at it:
                        //
                        //   error[E0277]: the trait bound `bool: Bar` is not satisfied
                        //     --> $DIR/point-at-type-on-obligation-failure-2.rs:8:5
                        //      |
                        //      | trait Foo where <Self as Foo>>::Assoc: Bar {
                        //      |                 -------------------------- restricted in this bound
                        //   LL |     type Assoc;
                        //      |          ----- associated type defined here
                        //   ...
                        //   LL | impl Foo for () {
                        //      | --------------- in this `impl` item
                        //   LL |     type Assoc = bool;
                        //      |     ^^^^^^^^^^^^^^^^^^ the trait `Bar` is not implemented for `bool`
                        if let (
                            ty::Projection(ty::ProjectionTy { item_def_id, .. }),
                            Some(hir::ItemKind::Impl { items, .. }),
                        ) = (&proj.skip_binder().self_ty().kind, item.map(|i| &i.kind))
                        {
                            if let Some((impl_item, trait_assoc_item)) = trait_assoc_items
                                .iter()
                                .find(|i| i.def_id == *item_def_id)
                                .and_then(|trait_assoc_item| {
                                    items
                                        .iter()
                                        .find(|i| i.ident == trait_assoc_item.ident)
                                        .map(|impl_item| (impl_item, trait_assoc_item))
                                })
                            {
                                let bounds = trait_generics
                                    .map(|generics| {
                                        get_generic_bound_spans(
                                            &generics,
                                            trait_name,
                                            trait_assoc_item.ident,
                                        )
                                    })
                                    .unwrap_or_else(Vec::new);
                                cause.span = impl_item.span;
                                cause.code = traits::AssocTypeBound(Box::new(AssocTypeBoundData {
                                    impl_span: item_span,
                                    original: trait_assoc_item.ident.span,
                                    bounds,
                                }));
                            }
                        }
                    }
                    _ => {}
                }
            };

        if let Elaborate::All = elaborate {
            // FIXME: Make `extend_cause_with_original_assoc_item_obligation` take an iterator
            // instead of a slice.
            let trait_assoc_items: Vec<_> =
                tcx.associated_items(trait_ref.def_id).in_definition_order().copied().collect();

            let predicates = obligations.iter().map(|obligation| obligation.predicate).collect();
            let implied_obligations = traits::elaborate_predicates(tcx, predicates);
            let implied_obligations = implied_obligations.map(|pred| {
                let mut cause = cause.clone();
                extend_cause_with_original_assoc_item_obligation(
                    &mut cause,
                    &pred,
                    &*trait_assoc_items,
                );
                traits::Obligation::new(cause, param_env, pred)
            });
            self.out.extend(implied_obligations);
        }

        self.out.extend(obligations);

        self.out.extend(trait_ref.substs.types().filter(|ty| !ty.has_escaping_bound_vars()).map(
            |ty| traits::Obligation::new(cause.clone(), param_env, ty::Predicate::WellFormed(ty)),
        ));
    }

    /// Pushes the obligations required for `trait_ref::Item` to be WF
    /// into `self.out`.
    fn compute_projection(&mut self, data: ty::ProjectionTy<'tcx>) {
        // A projection is well-formed if (a) the trait ref itself is
        // WF and (b) the trait-ref holds.  (It may also be
        // normalizable and be WF that way.)
        let trait_ref = data.trait_ref(self.infcx.tcx);
        self.compute_trait_ref(&trait_ref, Elaborate::None);

        if !data.has_escaping_bound_vars() {
            let predicate = trait_ref.without_const().to_predicate();
            let cause = self.cause(traits::ProjectionWf(data));
            self.out.push(traits::Obligation::new(cause, self.param_env, predicate));
        }
    }

    /// Pushes the obligations required for an array length to be WF
    /// into `self.out`.
    fn compute_array_len(&mut self, constant: ty::Const<'tcx>) {
        if let ty::ConstKind::Unevaluated(def_id, substs, promoted) = constant.val {
            assert!(promoted.is_none());

            let obligations = self.nominal_obligations(def_id, substs);
            self.out.extend(obligations);

            let predicate = ty::Predicate::ConstEvaluatable(def_id, substs);
            let cause = self.cause(traits::MiscObligation);
            self.out.push(traits::Obligation::new(cause, self.param_env, predicate));
        }
    }

    fn require_sized(&mut self, subty: Ty<'tcx>, cause: traits::ObligationCauseCode<'tcx>) {
        if !subty.has_escaping_bound_vars() {
            let cause = self.cause(cause);
            let trait_ref = ty::TraitRef {
                def_id: self.infcx.tcx.require_lang_item(lang_items::SizedTraitLangItem, None),
                substs: self.infcx.tcx.mk_substs_trait(subty, &[]),
            };
            self.out.push(traits::Obligation::new(
                cause,
                self.param_env,
                trait_ref.without_const().to_predicate(),
            ));
        }
    }

    /// Pushes new obligations into `out`. Returns `true` if it was able
    /// to generate all the predicates needed to validate that `ty0`
    /// is WF. Returns false if `ty0` is an unresolved type variable,
    /// in which case we are not able to simplify at all.
    fn compute(&mut self, ty0: Ty<'tcx>) -> bool {
        let mut subtys = ty0.walk();
        let param_env = self.param_env;
        while let Some(ty) = subtys.next() {
            match ty.kind {
                ty::Bool
                | ty::Char
                | ty::Int(..)
                | ty::Uint(..)
                | ty::Float(..)
                | ty::Error
                | ty::Str
                | ty::GeneratorWitness(..)
                | ty::Never
                | ty::Param(_)
                | ty::Bound(..)
                | ty::Placeholder(..)
                | ty::Foreign(..) => {
                    // WfScalar, WfParameter, etc
                }

                ty::Slice(subty) => {
                    self.require_sized(subty, traits::SliceOrArrayElem);
                }

                ty::Array(subty, len) => {
                    self.require_sized(subty, traits::SliceOrArrayElem);
                    self.compute_array_len(*len);
                }

                ty::Tuple(ref tys) => {
                    if let Some((_last, rest)) = tys.split_last() {
                        for elem in rest {
                            self.require_sized(elem.expect_ty(), traits::TupleElem);
                        }
                    }
                }

                ty::RawPtr(_) => {
                    // simple cases that are WF if their type args are WF
                }

                ty::Projection(data) => {
                    subtys.skip_current_subtree(); // subtree handled by compute_projection
                    self.compute_projection(data);
                }

                ty::UnnormalizedProjection(..) => bug!("only used with chalk-engine"),

                ty::Adt(def, substs) => {
                    // WfNominalType
                    let obligations = self.nominal_obligations(def.did, substs);
                    self.out.extend(obligations);
                }

                ty::FnDef(did, substs) => {
                    let obligations = self.nominal_obligations(did, substs);
                    self.out.extend(obligations);
                }

                ty::Ref(r, rty, _) => {
                    // WfReference
                    if !r.has_escaping_bound_vars() && !rty.has_escaping_bound_vars() {
                        let cause = self.cause(traits::ReferenceOutlivesReferent(ty));
                        self.out.push(traits::Obligation::new(
                            cause,
                            param_env,
                            ty::Predicate::TypeOutlives(ty::Binder::dummy(ty::OutlivesPredicate(
                                rty, r,
                            ))),
                        ));
                    }
                }

                ty::Generator(..) => {
                    // Walk ALL the types in the generator: this will
                    // include the upvar types as well as the yield
                    // type. Note that this is mildly distinct from
                    // the closure case, where we have to be careful
                    // about the signature of the closure. We don't
                    // have the problem of implied bounds here since
                    // generators don't take arguments.
                }

                ty::Closure(def_id, substs) => {
                    // Only check the upvar types for WF, not the rest
                    // of the types within. This is needed because we
                    // capture the signature and it may not be WF
                    // without the implied bounds. Consider a closure
                    // like `|x: &'a T|` -- it may be that `T: 'a` is
                    // not known to hold in the creator's context (and
                    // indeed the closure may not be invoked by its
                    // creator, but rather turned to someone who *can*
                    // verify that).
                    //
                    // The special treatment of closures here really
                    // ought not to be necessary either; the problem
                    // is related to #25860 -- there is no way for us
                    // to express a fn type complete with the implied
                    // bounds that it is assuming. I think in reality
                    // the WF rules around fn are a bit messed up, and
                    // that is the rot problem: `fn(&'a T)` should
                    // probably always be WF, because it should be
                    // shorthand for something like `where(T: 'a) {
                    // fn(&'a T) }`, as discussed in #25860.
                    //
                    // Note that we are also skipping the generic
                    // types. This is consistent with the `outlives`
                    // code, but anyway doesn't matter: within the fn
                    // body where they are created, the generics will
                    // always be WF, and outside of that fn body we
                    // are not directly inspecting closure types
                    // anyway, except via auto trait matching (which
                    // only inspects the upvar types).
                    subtys.skip_current_subtree(); // subtree handled by compute_projection
                    for upvar_ty in substs.as_closure().upvar_tys(def_id, self.infcx.tcx) {
                        self.compute(upvar_ty);
                    }
                }

                ty::FnPtr(_) => {
                    // let the loop iterate into the argument/return
                    // types appearing in the fn signature
                }

                ty::Opaque(did, substs) => {
                    // all of the requirements on type parameters
                    // should've been checked by the instantiation
                    // of whatever returned this exact `impl Trait`.

                    // for named opaque `impl Trait` types we still need to check them
                    if ty::is_impl_trait_defn(self.infcx.tcx, did).is_none() {
                        let obligations = self.nominal_obligations(did, substs);
                        self.out.extend(obligations);
                    }
                }

                ty::Dynamic(data, r) => {
                    // WfObject
                    //
                    // Here, we defer WF checking due to higher-ranked
                    // regions. This is perhaps not ideal.
                    self.from_object_ty(ty, data, r);

                    // FIXME(#27579) RFC also considers adding trait
                    // obligations that don't refer to Self and
                    // checking those

                    let defer_to_coercion = self.infcx.tcx.features().object_safe_for_dispatch;

                    if !defer_to_coercion {
                        let cause = self.cause(traits::MiscObligation);
                        let component_traits = data.auto_traits().chain(data.principal_def_id());
                        self.out.extend(component_traits.map(|did| {
                            traits::Obligation::new(
                                cause.clone(),
                                param_env,
                                ty::Predicate::ObjectSafe(did),
                            )
                        }));
                    }
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
                ty::Infer(_) => {
                    let ty = self.infcx.shallow_resolve(ty);
                    if let ty::Infer(_) = ty.kind {
                        // not yet resolved...
                        if ty == ty0 {
                            // ...this is the type we started from! no progress.
                            return false;
                        }

                        let cause = self.cause(traits::MiscObligation);
                        self.out.push(
                            // ...not the type we started from, so we made progress.
                            traits::Obligation::new(
                                cause,
                                self.param_env,
                                ty::Predicate::WellFormed(ty),
                            ),
                        );
                    } else {
                        // Yes, resolved, proceed with the
                        // result. Should never return false because
                        // `ty` is not a Infer.
                        assert!(self.compute(ty));
                    }
                }
            }
        }

        // if we made it through that loop above, we made progress!
        return true;
    }

    fn nominal_obligations(
        &mut self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        let predicates = self.infcx.tcx.predicates_of(def_id).instantiate(self.infcx.tcx, substs);
        let cause = self.cause(traits::ItemObligation(def_id));
        predicates
            .predicates
            .into_iter()
            .map(|pred| traits::Obligation::new(cause.clone(), self.param_env, pred))
            .filter(|pred| !pred.has_escaping_bound_vars())
            .collect()
    }

    fn from_object_ty(
        &mut self,
        ty: Ty<'tcx>,
        data: ty::Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>,
        region: ty::Region<'tcx>,
    ) {
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
        if !data.has_escaping_bound_vars() && !region.has_escaping_bound_vars() {
            let implicit_bounds = object_region_bounds(self.infcx.tcx, data);

            let explicit_bound = region;

            self.out.reserve(implicit_bounds.len());
            for implicit_bound in implicit_bounds {
                let cause = self.cause(traits::ObjectTypeBound(ty, explicit_bound));
                let outlives =
                    ty::Binder::dummy(ty::OutlivesPredicate(explicit_bound, implicit_bound));
                self.out.push(traits::Obligation::new(
                    cause,
                    self.param_env,
                    outlives.to_predicate(),
                ));
            }
        }
    }
}

/// Given an object type like `SomeTrait + Send`, computes the lifetime
/// bounds that must hold on the elided self type. These are derived
/// from the declarations of `SomeTrait`, `Send`, and friends -- if
/// they declare `trait SomeTrait : 'static`, for example, then
/// `'static` would appear in the list. The hard work is done by
/// `infer::required_region_bounds`, see that for more information.
pub fn object_region_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    existential_predicates: ty::Binder<&'tcx ty::List<ty::ExistentialPredicate<'tcx>>>,
) -> Vec<ty::Region<'tcx>> {
    // Since we don't actually *know* the self type for an object,
    // this "open(err)" serves as a kind of dummy standin -- basically
    // a placeholder type.
    let open_ty = tcx.mk_ty_infer(ty::FreshTy(0));

    let predicates = existential_predicates
        .iter()
        .filter_map(|predicate| {
            if let ty::ExistentialPredicate::Projection(_) = *predicate.skip_binder() {
                None
            } else {
                Some(predicate.with_self_ty(tcx, open_ty))
            }
        })
        .collect();

    required_region_bounds(tcx, open_ty, predicates)
}

/// Find the span of a generic bound affecting an associated type.
fn get_generic_bound_spans(
    generics: &hir::Generics<'_>,
    trait_name: Option<&Ident>,
    assoc_item_name: Ident,
) -> Vec<Span> {
    let mut bounds = vec![];
    for clause in generics.where_clause.predicates.iter() {
        if let hir::WherePredicate::BoundPredicate(pred) = clause {
            match &pred.bounded_ty.kind {
                hir::TyKind::Path(hir::QPath::Resolved(Some(ty), path)) => {
                    let mut s = path.segments.iter();
                    if let (a, Some(b), None) = (s.next(), s.next(), s.next()) {
                        if a.map(|s| &s.ident) == trait_name
                            && b.ident == assoc_item_name
                            && is_self_path(&ty.kind)
                        {
                            // `<Self as Foo>::Bar`
                            bounds.push(pred.span);
                        }
                    }
                }
                hir::TyKind::Path(hir::QPath::TypeRelative(ty, segment)) => {
                    if segment.ident == assoc_item_name {
                        if is_self_path(&ty.kind) {
                            // `Self::Bar`
                            bounds.push(pred.span);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    bounds
}

fn is_self_path(kind: &hir::TyKind<'_>) -> bool {
    match kind {
        hir::TyKind::Path(hir::QPath::Resolved(None, path)) => {
            let mut s = path.segments.iter();
            if let (Some(segment), None) = (s.next(), s.next()) {
                if segment.ident.name == kw::SelfUpper {
                    // `type(Self)`
                    return true;
                }
            }
        }
        _ => {}
    }
    false
}
