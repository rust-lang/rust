use crate::infer::InferCtxt;
use crate::opaque_types::required_region_bounds;
use crate::traits;
use rustc_data_structures::sync::Lrc;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt, TypeFoldable, WithConstness};
use rustc_span::Span;

use std::iter;
/// Returns the set of obligations needed to make `arg` well-formed.
/// If `arg` contains unresolved inference variables, this may include
/// further WF obligations. However, if `arg` IS an unresolved
/// inference variable, returns `None`, because we are not able to
/// make any progress at all. This is to prevent "livelock" where we
/// say "$0 is WF if $0 is WF".
pub fn obligations<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
    recursion_depth: usize,
    arg: GenericArg<'tcx>,
    span: Span,
) -> Option<Vec<traits::PredicateObligation<'tcx>>> {
    // Handle the "livelock" case (see comment above) by bailing out if necessary.
    let arg = match arg.unpack() {
        GenericArgKind::Type(ty) => {
            match ty.kind() {
                ty::Infer(ty::TyVar(_)) => {
                    let resolved_ty = infcx.shallow_resolve(ty);
                    if resolved_ty == ty {
                        // No progress, bail out to prevent "livelock".
                        return None;
                    }

                    resolved_ty
                }
                _ => ty,
            }
            .into()
        }
        GenericArgKind::Const(ct) => {
            match ct.val {
                ty::ConstKind::Infer(infer) => {
                    let resolved = infcx.shallow_resolve(infer);
                    if resolved == infer {
                        // No progress.
                        return None;
                    }

                    infcx.tcx.mk_const(ty::Const { val: ty::ConstKind::Infer(resolved), ty: ct.ty })
                }
                _ => ct,
            }
            .into()
        }
        // There is nothing we have to do for lifetimes.
        GenericArgKind::Lifetime(..) => return Some(Vec::new()),
    };

    let mut wf =
        WfPredicates { infcx, param_env, body_id, span, out: vec![], recursion_depth, item: None };
    wf.compute(arg);
    debug!("wf::obligations({:?}, body_id={:?}) = {:?}", arg, body_id, wf.out);

    let result = wf.normalize();
    debug!("wf::obligations({:?}, body_id={:?}) ~~> {:?}", arg, body_id, result);
    Some(result)
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
    let mut wf =
        WfPredicates { infcx, param_env, body_id, span, out: vec![], recursion_depth: 0, item };
    wf.compute_trait_ref(trait_ref, Elaborate::All);
    debug!(obligations = ?wf.out);
    wf.normalize()
}

pub fn predicate_obligations<'a, 'tcx>(
    infcx: &InferCtxt<'a, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: hir::HirId,
    predicate: ty::Predicate<'tcx>,
    span: Span,
) -> Vec<traits::PredicateObligation<'tcx>> {
    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id,
        span,
        out: vec![],
        recursion_depth: 0,
        item: None,
    };

    // It's ok to skip the binder here because wf code is prepared for it
    match predicate.kind().skip_binder() {
        ty::PredicateKind::Trait(t) => {
            wf.compute_trait_ref(&t.trait_ref, Elaborate::None);
        }
        ty::PredicateKind::RegionOutlives(..) => {}
        ty::PredicateKind::TypeOutlives(ty::OutlivesPredicate(ty, _reg)) => {
            wf.compute(ty.into());
        }
        ty::PredicateKind::Projection(t) => {
            wf.compute_projection(t.projection_ty);
            wf.compute(t.ty.into());
        }
        ty::PredicateKind::WellFormed(arg) => {
            wf.compute(arg);
        }
        ty::PredicateKind::ObjectSafe(_) => {}
        ty::PredicateKind::ClosureKind(..) => {}
        ty::PredicateKind::Subtype(ty::SubtypePredicate { a, b, a_is_expected: _ }) => {
            wf.compute(a.into());
            wf.compute(b.into());
        }
        ty::PredicateKind::Coerce(ty::CoercePredicate { a, b }) => {
            wf.compute(a.into());
            wf.compute(b.into());
        }
        ty::PredicateKind::ConstEvaluatable(uv) => {
            let substs = uv.substs(wf.tcx());
            let obligations = wf.nominal_obligations(uv.def.did, substs);
            wf.out.extend(obligations);

            for arg in substs.iter() {
                wf.compute(arg);
            }
        }
        ty::PredicateKind::ConstEquate(c1, c2) => {
            wf.compute(c1.into());
            wf.compute(c2.into());
        }
        ty::PredicateKind::TypeWellFormedFromEnv(..) => {
            bug!("TypeWellFormedFromEnv is only used for Chalk")
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
    recursion_depth: usize,
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

fn extend_cause_with_original_assoc_item_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    trait_ref: &ty::TraitRef<'tcx>,
    item: Option<&hir::Item<'tcx>>,
    cause: &mut traits::ObligationCause<'tcx>,
    pred: &ty::Predicate<'tcx>,
    mut trait_assoc_items: impl Iterator<Item = &'tcx ty::AssocItem>,
) {
    debug!(
        "extended_cause_with_original_assoc_item_obligation {:?} {:?} {:?} {:?}",
        trait_ref, item, cause, pred
    );
    let items = match item {
        Some(hir::Item { kind: hir::ItemKind::Impl(impl_), .. }) => impl_.items,
        _ => return,
    };
    let fix_span =
        |impl_item_ref: &hir::ImplItemRef| match tcx.hir().impl_item(impl_item_ref.id).kind {
            hir::ImplItemKind::Const(ty, _) | hir::ImplItemKind::TyAlias(ty) => ty.span,
            _ => impl_item_ref.span,
        };

    // It is fine to skip the binder as we don't care about regions here.
    match pred.kind().skip_binder() {
        ty::PredicateKind::Projection(proj) => {
            // The obligation comes not from the current `impl` nor the `trait` being implemented,
            // but rather from a "second order" obligation, where an associated type has a
            // projection coming from another associated type. See
            // `src/test/ui/associated-types/point-at-type-on-obligation-failure.rs` and
            // `traits-assoc-type-in-supertrait-bad.rs`.
            if let ty::Projection(projection_ty) = proj.ty.kind() {
                let trait_assoc_item = tcx.associated_item(projection_ty.item_def_id);
                if let Some(impl_item_span) =
                    items.iter().find(|item| item.ident == trait_assoc_item.ident).map(fix_span)
                {
                    cause.make_mut().span = impl_item_span;
                }
            }
        }
        ty::PredicateKind::Trait(pred) => {
            // An associated item obligation born out of the `trait` failed to be met. An example
            // can be seen in `ui/associated-types/point-at-type-on-obligation-failure-2.rs`.
            debug!("extended_cause_with_original_assoc_item_obligation trait proj {:?}", pred);
            if let ty::Projection(ty::ProjectionTy { item_def_id, .. }) = *pred.self_ty().kind() {
                if let Some(impl_item_span) = trait_assoc_items
                    .find(|i| i.def_id == item_def_id)
                    .and_then(|trait_assoc_item| {
                        items.iter().find(|i| i.ident == trait_assoc_item.ident).map(fix_span)
                    })
                {
                    cause.make_mut().span = impl_item_span;
                }
            }
        }
        _ => {}
    }
}

impl<'a, 'tcx> WfPredicates<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn cause(&self, code: traits::ObligationCauseCode<'tcx>) -> traits::ObligationCause<'tcx> {
        traits::ObligationCause::new(self.span, self.body_id, code)
    }

    fn normalize(mut self) -> Vec<traits::PredicateObligation<'tcx>> {
        let cause = self.cause(traits::MiscObligation);
        let infcx = &mut self.infcx;
        let param_env = self.param_env;
        let mut obligations = Vec::with_capacity(self.out.len());
        for mut obligation in self.out {
            assert!(!obligation.has_escaping_bound_vars());
            let mut selcx = traits::SelectionContext::new(infcx);
            // Don't normalize the whole obligation, the param env is either
            // already normalized, or we're currently normalizing the
            // param_env. Either way we should only normalize the predicate.
            let normalized_predicate = traits::project::normalize_with_depth_to(
                &mut selcx,
                param_env,
                cause.clone(),
                self.recursion_depth,
                obligation.predicate,
                &mut obligations,
            );
            obligation.predicate = normalized_predicate;
            obligations.push(obligation);
        }
        obligations
    }

    /// Pushes the obligations required for `trait_ref` to be WF into `self.out`.
    fn compute_trait_ref(&mut self, trait_ref: &ty::TraitRef<'tcx>, elaborate: Elaborate) {
        let tcx = self.infcx.tcx;
        let obligations = self.nominal_obligations(trait_ref.def_id, trait_ref.substs);

        debug!("compute_trait_ref obligations {:?}", obligations);
        let cause = self.cause(traits::MiscObligation);
        let param_env = self.param_env;
        let depth = self.recursion_depth;

        let item = self.item;

        let extend = |obligation: traits::PredicateObligation<'tcx>| {
            let mut cause = cause.clone();
            if let Some(parent_trait_ref) = obligation.predicate.to_opt_poly_trait_ref() {
                let derived_cause = traits::DerivedObligationCause {
                    parent_trait_ref: parent_trait_ref.value,
                    parent_code: Lrc::new(obligation.cause.code.clone()),
                };
                cause.make_mut().code =
                    traits::ObligationCauseCode::DerivedObligation(derived_cause);
            }
            extend_cause_with_original_assoc_item_obligation(
                tcx,
                trait_ref,
                item,
                &mut cause,
                &obligation.predicate,
                tcx.associated_items(trait_ref.def_id).in_definition_order(),
            );
            traits::Obligation::with_depth(cause, depth, param_env, obligation.predicate)
        };

        if let Elaborate::All = elaborate {
            let implied_obligations = traits::util::elaborate_obligations(tcx, obligations);
            let implied_obligations = implied_obligations.map(extend);
            self.out.extend(implied_obligations);
        } else {
            self.out.extend(obligations);
        }

        let tcx = self.tcx();
        self.out.extend(
            trait_ref
                .substs
                .iter()
                .enumerate()
                .filter(|(_, arg)| {
                    matches!(arg.unpack(), GenericArgKind::Type(..) | GenericArgKind::Const(..))
                })
                .filter(|(_, arg)| !arg.has_escaping_bound_vars())
                .map(|(i, arg)| {
                    let mut new_cause = cause.clone();
                    // The first subst is the self ty - use the correct span for it.
                    if i == 0 {
                        if let Some(hir::ItemKind::Impl(hir::Impl { self_ty, .. })) =
                            item.map(|i| &i.kind)
                        {
                            new_cause.make_mut().span = self_ty.span;
                        }
                    }
                    traits::Obligation::with_depth(
                        new_cause,
                        depth,
                        param_env,
                        ty::Binder::dummy(ty::PredicateKind::WellFormed(arg)).to_predicate(tcx),
                    )
                }),
        );
    }

    /// Pushes the obligations required for `trait_ref::Item` to be WF
    /// into `self.out`.
    fn compute_projection(&mut self, data: ty::ProjectionTy<'tcx>) {
        // A projection is well-formed if
        //
        // (a) its predicates hold (*)
        // (b) its substs are wf
        //
        // (*) The predicates of an associated type include the predicates of
        //     the trait that it's contained in. For example, given
        //
        // trait A<T>: Clone {
        //     type X where T: Copy;
        // }
        //
        // The predicates of `<() as A<i32>>::X` are:
        // [
        //     `(): Sized`
        //     `(): Clone`
        //     `(): A<i32>`
        //     `i32: Sized`
        //     `i32: Clone`
        //     `i32: Copy`
        // ]
        let obligations = self.nominal_obligations(data.item_def_id, data.substs);
        self.out.extend(obligations);

        let tcx = self.tcx();
        let cause = self.cause(traits::MiscObligation);
        let param_env = self.param_env;
        let depth = self.recursion_depth;

        self.out.extend(
            data.substs
                .iter()
                .filter(|arg| {
                    matches!(arg.unpack(), GenericArgKind::Type(..) | GenericArgKind::Const(..))
                })
                .filter(|arg| !arg.has_escaping_bound_vars())
                .map(|arg| {
                    traits::Obligation::with_depth(
                        cause.clone(),
                        depth,
                        param_env,
                        ty::Binder::dummy(ty::PredicateKind::WellFormed(arg)).to_predicate(tcx),
                    )
                }),
        );
    }

    fn require_sized(&mut self, subty: Ty<'tcx>, cause: traits::ObligationCauseCode<'tcx>) {
        if !subty.has_escaping_bound_vars() {
            let cause = self.cause(cause);
            let trait_ref = ty::TraitRef {
                def_id: self.infcx.tcx.require_lang_item(LangItem::Sized, None),
                substs: self.infcx.tcx.mk_substs_trait(subty, &[]),
            };
            self.out.push(traits::Obligation::with_depth(
                cause,
                self.recursion_depth,
                self.param_env,
                ty::Binder::dummy(trait_ref).without_const().to_predicate(self.infcx.tcx),
            ));
        }
    }

    /// Pushes all the predicates needed to validate that `ty` is WF into `out`.
    fn compute(&mut self, arg: GenericArg<'tcx>) {
        let mut walker = arg.walk(self.tcx());
        let param_env = self.param_env;
        let depth = self.recursion_depth;
        while let Some(arg) = walker.next() {
            let ty = match arg.unpack() {
                GenericArgKind::Type(ty) => ty,

                // No WF constraints for lifetimes being present, any outlives
                // obligations are handled by the parent (e.g. `ty::Ref`).
                GenericArgKind::Lifetime(_) => continue,

                GenericArgKind::Const(constant) => {
                    match constant.val {
                        ty::ConstKind::Unevaluated(uv) => {
                            assert!(uv.promoted.is_none());
                            let substs = uv.substs(self.tcx());

                            let obligations = self.nominal_obligations(uv.def.did, substs);
                            self.out.extend(obligations);

                            let predicate = ty::Binder::dummy(ty::PredicateKind::ConstEvaluatable(
                                ty::Unevaluated::new(uv.def, substs),
                            ))
                            .to_predicate(self.tcx());
                            let cause = self.cause(traits::MiscObligation);
                            self.out.push(traits::Obligation::with_depth(
                                cause,
                                self.recursion_depth,
                                self.param_env,
                                predicate,
                            ));
                        }
                        ty::ConstKind::Infer(infer) => {
                            let resolved = self.infcx.shallow_resolve(infer);
                            // the `InferConst` changed, meaning that we made progress.
                            if resolved != infer {
                                let cause = self.cause(traits::MiscObligation);

                                let resolved_constant = self.infcx.tcx.mk_const(ty::Const {
                                    val: ty::ConstKind::Infer(resolved),
                                    ..*constant
                                });
                                self.out.push(traits::Obligation::with_depth(
                                    cause,
                                    self.recursion_depth,
                                    self.param_env,
                                    ty::Binder::dummy(ty::PredicateKind::WellFormed(
                                        resolved_constant.into(),
                                    ))
                                    .to_predicate(self.tcx()),
                                ));
                            }
                        }
                        ty::ConstKind::Error(_)
                        | ty::ConstKind::Param(_)
                        | ty::ConstKind::Bound(..)
                        | ty::ConstKind::Placeholder(..) => {
                            // These variants are trivially WF, so nothing to do here.
                        }
                        ty::ConstKind::Value(..) => {
                            // FIXME: Enforce that values are structurally-matchable.
                        }
                    }
                    continue;
                }
            };

            match *ty.kind() {
                ty::Bool
                | ty::Char
                | ty::Int(..)
                | ty::Uint(..)
                | ty::Float(..)
                | ty::Error(_)
                | ty::Str
                | ty::GeneratorWitness(..)
                | ty::Never
                | ty::Param(_)
                | ty::Bound(..)
                | ty::Placeholder(..)
                | ty::Foreign(..) => {
                    // WfScalar, WfParameter, etc
                }

                // Can only infer to `ty::Int(_) | ty::Uint(_)`.
                ty::Infer(ty::IntVar(_)) => {}

                // Can only infer to `ty::Float(_)`.
                ty::Infer(ty::FloatVar(_)) => {}

                ty::Slice(subty) => {
                    self.require_sized(subty, traits::SliceOrArrayElem);
                }

                ty::Array(subty, _) => {
                    self.require_sized(subty, traits::SliceOrArrayElem);
                    // Note that we handle the len is implicitly checked while walking `arg`.
                }

                ty::Tuple(ref tys) => {
                    if let Some((_last, rest)) = tys.split_last() {
                        for elem in rest {
                            self.require_sized(elem.expect_ty(), traits::TupleElem);
                        }
                    }
                }

                ty::RawPtr(_) => {
                    // Simple cases that are WF if their type args are WF.
                }

                ty::Projection(data) => {
                    walker.skip_current_subtree(); // Subtree handled by compute_projection.
                    self.compute_projection(data);
                }

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
                        self.out.push(traits::Obligation::with_depth(
                            cause,
                            depth,
                            param_env,
                            ty::Binder::dummy(ty::PredicateKind::TypeOutlives(
                                ty::OutlivesPredicate(rty, r),
                            ))
                            .to_predicate(self.tcx()),
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

                ty::Closure(_, substs) => {
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
                    walker.skip_current_subtree(); // subtree handled below
                    // FIXME(eddyb) add the type to `walker` instead of recursing.
                    self.compute(substs.as_closure().tupled_upvars_ty().into());
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

                    let defer_to_coercion = self.tcx().features().object_safe_for_dispatch;

                    if !defer_to_coercion {
                        let cause = self.cause(traits::MiscObligation);
                        let component_traits = data.auto_traits().chain(data.principal_def_id());
                        let tcx = self.tcx();
                        self.out.extend(component_traits.map(|did| {
                            traits::Obligation::with_depth(
                                cause.clone(),
                                depth,
                                param_env,
                                ty::Binder::dummy(ty::PredicateKind::ObjectSafe(did))
                                    .to_predicate(tcx),
                            )
                        }));
                    }
                }

                // Inference variables are the complicated case, since we don't
                // know what type they are. We do two things:
                //
                // 1. Check if they have been resolved, and if so proceed with
                //    THAT type.
                // 2. If not, we've at least simplified things (e.g., we went
                //    from `Vec<$0>: WF` to `$0: WF`), so we can
                //    register a pending obligation and keep
                //    moving. (Goal is that an "inductive hypothesis"
                //    is satisfied to ensure termination.)
                // See also the comment on `fn obligations`, describing "livelock"
                // prevention, which happens before this can be reached.
                ty::Infer(_) => {
                    let ty = self.infcx.shallow_resolve(ty);
                    if let ty::Infer(ty::TyVar(_)) = ty.kind() {
                        // Not yet resolved, but we've made progress.
                        let cause = self.cause(traits::MiscObligation);
                        self.out.push(traits::Obligation::with_depth(
                            cause,
                            self.recursion_depth,
                            param_env,
                            ty::Binder::dummy(ty::PredicateKind::WellFormed(ty.into()))
                                .to_predicate(self.tcx()),
                        ));
                    } else {
                        // Yes, resolved, proceed with the result.
                        // FIXME(eddyb) add the type to `walker` instead of recursing.
                        self.compute(ty.into());
                    }
                }
            }
        }
    }

    fn nominal_obligations(
        &mut self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        let predicates = self.infcx.tcx.predicates_of(def_id);
        let mut origins = vec![def_id; predicates.predicates.len()];
        let mut head = predicates;
        while let Some(parent) = head.parent {
            head = self.infcx.tcx.predicates_of(parent);
            origins.extend(iter::repeat(parent).take(head.predicates.len()));
        }

        let predicates = predicates.instantiate(self.infcx.tcx, substs);
        debug_assert_eq!(predicates.predicates.len(), origins.len());

        iter::zip(iter::zip(predicates.predicates, predicates.spans), origins.into_iter().rev())
            .map(|((pred, span), origin_def_id)| {
                let code = if span.is_dummy() {
                    traits::MiscObligation
                } else {
                    traits::BindingObligation(origin_def_id, span)
                };
                let cause = self.cause(code);
                traits::Obligation::with_depth(cause, self.recursion_depth, self.param_env, pred)
            })
            .filter(|pred| !pred.has_escaping_bound_vars())
            .collect()
    }

    fn from_object_ty(
        &mut self,
        ty: Ty<'tcx>,
        data: &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
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
                self.out.push(traits::Obligation::with_depth(
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    outlives.to_predicate(self.infcx.tcx),
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
    existential_predicates: &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
) -> Vec<ty::Region<'tcx>> {
    // Since we don't actually *know* the self type for an object,
    // this "open(err)" serves as a kind of dummy standin -- basically
    // a placeholder type.
    let open_ty = tcx.mk_ty_infer(ty::FreshTy(0));

    let predicates = existential_predicates.iter().filter_map(|predicate| {
        if let ty::ExistentialPredicate::Projection(_) = predicate.skip_binder() {
            None
        } else {
            Some(predicate.with_self_ty(tcx, open_ty))
        }
    });

    required_region_bounds(tcx, open_ty, predicates)
}
