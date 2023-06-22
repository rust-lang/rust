use crate::infer::InferCtxt;
use crate::traits;
use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};
use rustc_span::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_span::{Span, DUMMY_SP};

use std::iter;
/// Returns the set of obligations needed to make `arg` well-formed.
/// If `arg` contains unresolved inference variables, this may include
/// further WF obligations. However, if `arg` IS an unresolved
/// inference variable, returns `None`, because we are not able to
/// make any progress at all. This is to prevent "livelock" where we
/// say "$0 is WF if $0 is WF".
pub fn obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
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
                    } else {
                        resolved_ty
                    }
                }
                _ => ty,
            }
            .into()
        }
        GenericArgKind::Const(ct) => {
            match ct.kind() {
                ty::ConstKind::Infer(_) => {
                    let resolved = infcx.shallow_resolve(ct);
                    if resolved == ct {
                        // No progress.
                        return None;
                    } else {
                        resolved
                    }
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

    let result = wf.normalize(infcx);
    debug!("wf::obligations({:?}, body_id={:?}) ~~> {:?}", arg, body_id, result);
    Some(result)
}

/// Compute the predicates that are required for a type to be well-formed.
///
/// This is only intended to be used in the new solver, since it does not
/// take into account recursion depth or proper error-reporting spans.
pub fn unnormalized_obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    arg: GenericArg<'tcx>,
) -> Option<Vec<traits::PredicateObligation<'tcx>>> {
    debug_assert_eq!(arg, infcx.resolve_vars_if_possible(arg));

    // However, if `arg` IS an unresolved inference variable, returns `None`,
    // because we are not able to make any progress at all. This is to prevent
    // "livelock" where we say "$0 is WF if $0 is WF".
    if arg.is_non_region_infer() {
        return None;
    }

    if let ty::GenericArgKind::Lifetime(..) = arg.unpack() {
        return Some(vec![]);
    }

    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id: CRATE_DEF_ID,
        span: DUMMY_SP,
        out: vec![],
        recursion_depth: 0,
        item: None,
    };
    wf.compute(arg);
    Some(wf.out)
}

/// Returns the obligations that make this trait reference
/// well-formed. For example, if there is a trait `Set` defined like
/// `trait Set<K:Eq>`, then the trait reference `Foo: Set<Bar>` is WF
/// if `Bar: Eq`.
pub fn trait_obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    trait_pred: &ty::TraitPredicate<'tcx>,
    span: Span,
    item: &'tcx hir::Item<'tcx>,
) -> Vec<traits::PredicateObligation<'tcx>> {
    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id,
        span,
        out: vec![],
        recursion_depth: 0,
        item: Some(item),
    };
    wf.compute_trait_pred(trait_pred, Elaborate::All);
    debug!(obligations = ?wf.out);
    wf.normalize(infcx)
}

#[instrument(skip(infcx), ret)]
pub fn predicate_obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
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
        ty::PredicateKind::Clause(ty::ClauseKind::Trait(t)) => {
            wf.compute_trait_pred(&t, Elaborate::None);
        }
        ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..)) => {}
        ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
            ty,
            _reg,
        ))) => {
            wf.compute(ty.into());
        }
        ty::PredicateKind::Clause(ty::ClauseKind::Projection(t)) => {
            wf.compute_projection(t.projection_ty);
            wf.compute(match t.term.unpack() {
                ty::TermKind::Ty(ty) => ty.into(),
                ty::TermKind::Const(c) => c.into(),
            })
        }
        ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(ct, ty)) => {
            wf.compute(ct.into());
            wf.compute(ty.into());
        }
        ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(arg)) => {
            wf.compute(arg);
        }

        ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(ct)) => {
            wf.compute(ct.into());
        }

        ty::PredicateKind::ObjectSafe(_)
        | ty::PredicateKind::ClosureKind(..)
        | ty::PredicateKind::Subtype(..)
        | ty::PredicateKind::Coerce(..)
        | ty::PredicateKind::ConstEquate(..)
        | ty::PredicateKind::Ambiguous
        | ty::PredicateKind::AliasRelate(..)
        | ty::PredicateKind::TypeWellFormedFromEnv(..) => {
            bug!("We should only wf check where clauses, unexpected predicate: {predicate:?}")
        }
    }

    wf.normalize(infcx)
}

struct WfPredicates<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    span: Span,
    out: Vec<traits::PredicateObligation<'tcx>>,
    recursion_depth: usize,
    item: Option<&'tcx hir::Item<'tcx>>,
}

/// Controls whether we "elaborate" supertraits and so forth on the WF
/// predicates. This is a kind of hack to address #43784. The
/// underlying problem in that issue was a trait structure like:
///
/// ```ignore (illustrative)
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
    pred: ty::Predicate<'tcx>,
) {
    debug!(
        "extended_cause_with_original_assoc_item_obligation {:?} {:?} {:?} {:?}",
        trait_ref, item, cause, pred
    );
    let (items, impl_def_id) = match item {
        Some(hir::Item { kind: hir::ItemKind::Impl(impl_), owner_id, .. }) => {
            (impl_.items, *owner_id)
        }
        _ => return,
    };
    let fix_span =
        |impl_item_ref: &hir::ImplItemRef| match tcx.hir().impl_item(impl_item_ref.id).kind {
            hir::ImplItemKind::Const(ty, _) | hir::ImplItemKind::Type(ty) => ty.span,
            _ => impl_item_ref.span,
        };

    // It is fine to skip the binder as we don't care about regions here.
    match pred.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj)) => {
            // The obligation comes not from the current `impl` nor the `trait` being implemented,
            // but rather from a "second order" obligation, where an associated type has a
            // projection coming from another associated type. See
            // `tests/ui/associated-types/point-at-type-on-obligation-failure.rs` and
            // `traits-assoc-type-in-supertrait-bad.rs`.
            if let Some(ty::Alias(ty::Projection, projection_ty)) = proj.term.ty().map(|ty| ty.kind())
                && let Some(&impl_item_id) =
                    tcx.impl_item_implementor_ids(impl_def_id).get(&projection_ty.def_id)
                && let Some(impl_item_span) = items
                    .iter()
                    .find(|item| item.id.owner_id.to_def_id() == impl_item_id)
                    .map(fix_span)
            {
                cause.span = impl_item_span;
            }
        }
        ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
            // An associated item obligation born out of the `trait` failed to be met. An example
            // can be seen in `ui/associated-types/point-at-type-on-obligation-failure-2.rs`.
            debug!("extended_cause_with_original_assoc_item_obligation trait proj {:?}", pred);
            if let ty::Alias(ty::Projection, ty::AliasTy { def_id, .. }) = *pred.self_ty().kind()
                && let Some(&impl_item_id) =
                    tcx.impl_item_implementor_ids(impl_def_id).get(&def_id)
                && let Some(impl_item_span) = items
                    .iter()
                    .find(|item| item.id.owner_id.to_def_id() == impl_item_id)
                    .map(fix_span)
            {
                cause.span = impl_item_span;
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

    fn normalize(self, infcx: &InferCtxt<'tcx>) -> Vec<traits::PredicateObligation<'tcx>> {
        let cause = self.cause(traits::WellFormed(None));
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
    fn compute_trait_pred(&mut self, trait_pred: &ty::TraitPredicate<'tcx>, elaborate: Elaborate) {
        let tcx = self.tcx();
        let trait_ref = &trait_pred.trait_ref;

        // Negative trait predicates don't require supertraits to hold, just
        // that their substs are WF.
        if trait_pred.polarity == ty::ImplPolarity::Negative {
            self.compute_negative_trait_pred(trait_ref);
            return;
        }

        // if the trait predicate is not const, the wf obligations should not be const as well.
        let obligations = if trait_pred.constness == ty::BoundConstness::NotConst {
            self.nominal_obligations_without_const(trait_ref.def_id, trait_ref.substs)
        } else {
            self.nominal_obligations(trait_ref.def_id, trait_ref.substs)
        };

        debug!("compute_trait_pred obligations {:?}", obligations);
        let param_env = self.param_env;
        let depth = self.recursion_depth;

        let item = self.item;

        let extend = |traits::PredicateObligation { predicate, mut cause, .. }| {
            if let Some(parent_trait_pred) = predicate.to_opt_poly_trait_pred() {
                cause = cause.derived_cause(
                    parent_trait_pred,
                    traits::ObligationCauseCode::DerivedObligation,
                );
            }
            extend_cause_with_original_assoc_item_obligation(
                tcx, trait_ref, item, &mut cause, predicate,
            );
            traits::Obligation::with_depth(tcx, cause, depth, param_env, predicate)
        };

        if let Elaborate::All = elaborate {
            let implied_obligations = traits::util::elaborate(tcx, obligations);
            let implied_obligations = implied_obligations.map(extend);
            self.out.extend(implied_obligations);
        } else {
            self.out.extend(obligations);
        }

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
                    let mut cause = traits::ObligationCause::misc(self.span, self.body_id);
                    // The first subst is the self ty - use the correct span for it.
                    if i == 0 {
                        if let Some(hir::ItemKind::Impl(hir::Impl { self_ty, .. })) =
                            item.map(|i| &i.kind)
                        {
                            cause.span = self_ty.span;
                        }
                    }
                    traits::Obligation::with_depth(
                        tcx,
                        cause,
                        depth,
                        param_env,
                        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                            arg,
                        ))),
                    )
                }),
        );
    }

    // Compute the obligations that are required for `trait_ref` to be WF,
    // given that it is a *negative* trait predicate.
    fn compute_negative_trait_pred(&mut self, trait_ref: &ty::TraitRef<'tcx>) {
        for arg in trait_ref.substs {
            self.compute(arg);
        }
    }

    /// Pushes the obligations required for `trait_ref::Item` to be WF
    /// into `self.out`.
    fn compute_projection(&mut self, data: ty::AliasTy<'tcx>) {
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
        // Projection types do not require const predicates.
        let obligations = self.nominal_obligations_without_const(data.def_id, data.substs);
        self.out.extend(obligations);

        self.compute_projection_substs(data.substs);
    }

    fn compute_inherent_projection(&mut self, data: ty::AliasTy<'tcx>) {
        // An inherent projection is well-formed if
        //
        // (a) its predicates hold (*)
        // (b) its substs are wf
        //
        // (*) The predicates of an inherent associated type include the
        //     predicates of the impl that it's contained in.

        if !data.self_ty().has_escaping_bound_vars() {
            // FIXME(inherent_associated_types): Should this happen inside of a snapshot?
            // FIXME(inherent_associated_types): This is incompatible with the new solver and lazy norm!
            let substs = traits::project::compute_inherent_assoc_ty_substs(
                &mut traits::SelectionContext::new(self.infcx),
                self.param_env,
                data,
                self.cause(traits::WellFormed(None)),
                self.recursion_depth,
                &mut self.out,
            );
            // Inherent projection types do not require const predicates.
            let obligations = self.nominal_obligations_without_const(data.def_id, substs);
            self.out.extend(obligations);
        }

        self.compute_projection_substs(data.substs);
    }

    fn compute_projection_substs(&mut self, substs: SubstsRef<'tcx>) {
        let tcx = self.tcx();
        let cause = self.cause(traits::WellFormed(None));
        let param_env = self.param_env;
        let depth = self.recursion_depth;

        self.out.extend(
            substs
                .iter()
                .filter(|arg| {
                    matches!(arg.unpack(), GenericArgKind::Type(..) | GenericArgKind::Const(..))
                })
                .filter(|arg| !arg.has_escaping_bound_vars())
                .map(|arg| {
                    traits::Obligation::with_depth(
                        tcx,
                        cause.clone(),
                        depth,
                        param_env,
                        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                            arg,
                        ))),
                    )
                }),
        );
    }

    fn require_sized(&mut self, subty: Ty<'tcx>, cause: traits::ObligationCauseCode<'tcx>) {
        if !subty.has_escaping_bound_vars() {
            let cause = self.cause(cause);
            let trait_ref =
                ty::TraitRef::from_lang_item(self.tcx(), LangItem::Sized, cause.span, [subty]);
            self.out.push(traits::Obligation::with_depth(
                self.tcx(),
                cause,
                self.recursion_depth,
                self.param_env,
                ty::Binder::dummy(trait_ref).without_const(),
            ));
        }
    }

    /// Pushes all the predicates needed to validate that `ty` is WF into `out`.
    #[instrument(level = "debug", skip(self))]
    fn compute(&mut self, arg: GenericArg<'tcx>) {
        let mut walker = arg.walk();
        let param_env = self.param_env;
        let depth = self.recursion_depth;
        while let Some(arg) = walker.next() {
            debug!(?arg, ?self.out);
            let ty = match arg.unpack() {
                GenericArgKind::Type(ty) => ty,

                // No WF constraints for lifetimes being present, any outlives
                // obligations are handled by the parent (e.g. `ty::Ref`).
                GenericArgKind::Lifetime(_) => continue,

                GenericArgKind::Const(ct) => {
                    match ct.kind() {
                        ty::ConstKind::Unevaluated(uv) => {
                            if !ct.has_escaping_bound_vars() {
                                let obligations = self.nominal_obligations(uv.def, uv.substs);
                                self.out.extend(obligations);

                                let predicate = ty::Binder::dummy(ty::PredicateKind::Clause(
                                    ty::ClauseKind::ConstEvaluatable(ct),
                                ));
                                let cause = self.cause(traits::WellFormed(None));
                                self.out.push(traits::Obligation::with_depth(
                                    self.tcx(),
                                    cause,
                                    self.recursion_depth,
                                    self.param_env,
                                    predicate,
                                ));
                            }
                        }
                        ty::ConstKind::Infer(_) => {
                            let cause = self.cause(traits::WellFormed(None));

                            self.out.push(traits::Obligation::with_depth(
                                self.tcx(),
                                cause,
                                self.recursion_depth,
                                self.param_env,
                                ty::Binder::dummy(ty::PredicateKind::Clause(
                                    ty::ClauseKind::WellFormed(ct.into()),
                                )),
                            ));
                        }
                        ty::ConstKind::Expr(_) => {
                            // FIXME(generic_const_exprs): this doesnt verify that given `Expr(N + 1)` the
                            // trait bound `typeof(N): Add<typeof(1)>` holds. This is currently unnecessary
                            // as `ConstKind::Expr` is only produced via normalization of `ConstKind::Unevaluated`
                            // which means that the `DefId` would have been typeck'd elsewhere. However in
                            // the future we may allow directly lowering to `ConstKind::Expr` in which case
                            // we would not be proving bounds we should.

                            let predicate = ty::Binder::dummy(ty::PredicateKind::Clause(
                                ty::ClauseKind::ConstEvaluatable(ct),
                            ));
                            let cause = self.cause(traits::WellFormed(None));
                            self.out.push(traits::Obligation::with_depth(
                                self.tcx(),
                                cause,
                                self.recursion_depth,
                                self.param_env,
                                predicate,
                            ));
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

            debug!("wf bounds for ty={:?} ty.kind={:#?}", ty, ty.kind());

            match *ty.kind() {
                ty::Bool
                | ty::Char
                | ty::Int(..)
                | ty::Uint(..)
                | ty::Float(..)
                | ty::Error(_)
                | ty::Str
                | ty::GeneratorWitness(..)
                | ty::GeneratorWitnessMIR(..)
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
                        for &elem in rest {
                            self.require_sized(elem, traits::TupleElem);
                        }
                    }
                }

                ty::RawPtr(_) => {
                    // Simple cases that are WF if their type args are WF.
                }

                ty::Alias(ty::Projection, data) => {
                    walker.skip_current_subtree(); // Subtree handled by compute_projection.
                    self.compute_projection(data);
                }
                ty::Alias(ty::Inherent, data) => {
                    walker.skip_current_subtree(); // Subtree handled by compute_inherent_projection.
                    self.compute_inherent_projection(data);
                }

                ty::Adt(def, substs) => {
                    // WfNominalType
                    let obligations = self.nominal_obligations(def.did(), substs);
                    self.out.extend(obligations);
                }

                ty::FnDef(did, substs) => {
                    let obligations = self.nominal_obligations_without_const(did, substs);
                    self.out.extend(obligations);
                }

                ty::Ref(r, rty, _) => {
                    // WfReference
                    if !r.has_escaping_bound_vars() && !rty.has_escaping_bound_vars() {
                        let cause = self.cause(traits::ReferenceOutlivesReferent(ty));
                        self.out.push(traits::Obligation::with_depth(
                            self.tcx(),
                            cause,
                            depth,
                            param_env,
                            ty::Binder::dummy(ty::PredicateKind::Clause(
                                ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(rty, r)),
                            )),
                        ));
                    }
                }

                ty::Generator(did, substs, ..) => {
                    // Walk ALL the types in the generator: this will
                    // include the upvar types as well as the yield
                    // type. Note that this is mildly distinct from
                    // the closure case, where we have to be careful
                    // about the signature of the closure. We don't
                    // have the problem of implied bounds here since
                    // generators don't take arguments.
                    let obligations = self.nominal_obligations(did, substs);
                    self.out.extend(obligations);
                }

                ty::Closure(did, substs) => {
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
                    walker.skip_current_subtree(); // subtree handled below
                    // FIXME(eddyb) add the type to `walker` instead of recursing.
                    self.compute(substs.as_closure().tupled_upvars_ty().into());
                    // Note that we cannot skip the generic types
                    // types. Normally, within the fn
                    // body where they are created, the generics will
                    // always be WF, and outside of that fn body we
                    // are not directly inspecting closure types
                    // anyway, except via auto trait matching (which
                    // only inspects the upvar types).
                    // But when a closure is part of a type-alias-impl-trait
                    // then the function that created the defining site may
                    // have had more bounds available than the type alias
                    // specifies. This may cause us to have a closure in the
                    // hidden type that is not actually well formed and
                    // can cause compiler crashes when the user abuses unsafe
                    // code to procure such a closure.
                    // See tests/ui/type-alias-impl-trait/wf_check_closures.rs
                    let obligations = self.nominal_obligations(did, substs);
                    self.out.extend(obligations);
                }

                ty::FnPtr(_) => {
                    // let the loop iterate into the argument/return
                    // types appearing in the fn signature
                }

                ty::Alias(ty::Opaque, ty::AliasTy { def_id, substs, .. }) => {
                    // All of the requirements on type parameters
                    // have already been checked for `impl Trait` in
                    // return position. We do need to check type-alias-impl-trait though.
                    if self.tcx().is_type_alias_impl_trait(def_id) {
                        let obligations = self.nominal_obligations(def_id, substs);
                        self.out.extend(obligations);
                    }
                }

                ty::Alias(ty::Weak, ty::AliasTy { def_id, substs, .. }) => {
                    let obligations = self.nominal_obligations(def_id, substs);
                    self.out.extend(obligations);
                }

                ty::Dynamic(data, r, _) => {
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
                        let cause = self.cause(traits::WellFormed(None));
                        let component_traits = data.auto_traits().chain(data.principal_def_id());
                        let tcx = self.tcx();
                        self.out.extend(component_traits.map(|did| {
                            traits::Obligation::with_depth(
                                tcx,
                                cause.clone(),
                                depth,
                                param_env,
                                ty::Binder::dummy(ty::PredicateKind::ObjectSafe(did)),
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
                    let cause = self.cause(traits::WellFormed(None));
                    self.out.push(traits::Obligation::with_depth(
                        self.tcx(),
                        cause,
                        self.recursion_depth,
                        param_env,
                        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                            ty.into(),
                        ))),
                    ));
                }
            }

            debug!(?self.out);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn nominal_obligations_inner(
        &mut self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
        remap_constness: bool,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        let predicates = self.tcx().predicates_of(def_id);
        let mut origins = vec![def_id; predicates.predicates.len()];
        let mut head = predicates;
        while let Some(parent) = head.parent {
            head = self.tcx().predicates_of(parent);
            origins.extend(iter::repeat(parent).take(head.predicates.len()));
        }

        let predicates = predicates.instantiate(self.tcx(), substs);
        trace!("{:#?}", predicates);
        debug_assert_eq!(predicates.predicates.len(), origins.len());

        iter::zip(predicates, origins.into_iter().rev())
            .map(|((mut pred, span), origin_def_id)| {
                let code = if span.is_dummy() {
                    traits::ItemObligation(origin_def_id)
                } else {
                    traits::BindingObligation(origin_def_id, span)
                };
                let cause = self.cause(code);
                if remap_constness {
                    pred = pred.without_const(self.tcx());
                }
                traits::Obligation::with_depth(
                    self.tcx(),
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    pred,
                )
            })
            .filter(|pred| !pred.has_escaping_bound_vars())
            .collect()
    }

    fn nominal_obligations(
        &mut self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        self.nominal_obligations_inner(def_id, substs, false)
    }

    fn nominal_obligations_without_const(
        &mut self,
        def_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> Vec<traits::PredicateObligation<'tcx>> {
        self.nominal_obligations_inner(def_id, substs, true)
    }

    fn from_object_ty(
        &mut self,
        ty: Ty<'tcx>,
        data: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
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
            let implicit_bounds = object_region_bounds(self.tcx(), data);

            let explicit_bound = region;

            self.out.reserve(implicit_bounds.len());
            for implicit_bound in implicit_bounds {
                let cause = self.cause(traits::ObjectTypeBound(ty, explicit_bound));
                let outlives =
                    ty::Binder::dummy(ty::OutlivesPredicate(explicit_bound, implicit_bound));
                self.out.push(traits::Obligation::with_depth(
                    self.tcx(),
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    outlives,
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
    existential_predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> Vec<ty::Region<'tcx>> {
    // Since we don't actually *know* the self type for an object,
    // this "open(err)" serves as a kind of dummy standin -- basically
    // a placeholder type.
    let open_ty = tcx.mk_fresh_ty(0);

    let predicates = existential_predicates.iter().filter_map(|predicate| {
        if let ty::ExistentialPredicate::Projection(_) = predicate.skip_binder() {
            None
        } else {
            Some(predicate.with_self_ty(tcx, open_ty))
        }
    });

    required_region_bounds(tcx, open_ty, predicates)
}

/// Given a set of predicates that apply to an object type, returns
/// the region bounds that the (erased) `Self` type must
/// outlive. Precisely *because* the `Self` type is erased, the
/// parameter `erased_self_ty` must be supplied to indicate what type
/// has been used to represent `Self` in the predicates
/// themselves. This should really be a unique type; `FreshTy(0)` is a
/// popular choice.
///
/// N.B., in some cases, particularly around higher-ranked bounds,
/// this function returns a kind of conservative approximation.
/// That is, all regions returned by this function are definitely
/// required, but there may be other region bounds that are not
/// returned, as well as requirements like `for<'a> T: 'a`.
///
/// Requires that trait definitions have been processed so that we can
/// elaborate predicates and walk supertraits.
#[instrument(skip(tcx, predicates), level = "debug", ret)]
pub(crate) fn required_region_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    erased_self_ty: Ty<'tcx>,
    predicates: impl Iterator<Item = ty::Predicate<'tcx>>,
) -> Vec<ty::Region<'tcx>> {
    assert!(!erased_self_ty.has_escaping_bound_vars());

    traits::elaborate(tcx, predicates)
        .filter_map(|pred| {
            debug!(?pred);
            match pred.kind().skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::Trait(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(..))
                | ty::PredicateKind::Subtype(..)
                | ty::PredicateKind::Coerce(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(..))
                | ty::PredicateKind::ObjectSafe(..)
                | ty::PredicateKind::ClosureKind(..)
                | ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(..))
                | ty::PredicateKind::Clause(ty::ClauseKind::ConstEvaluatable(..))
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::Ambiguous
                | ty::PredicateKind::AliasRelate(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => None,
                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(
                    ref t,
                    ref r,
                ))) => {
                    // Search for a bound of the form `erased_self_ty
                    // : 'a`, but be wary of something like `for<'a>
                    // erased_self_ty : 'a` (we interpret a
                    // higher-ranked bound like that as 'static,
                    // though at present the code in `fulfill.rs`
                    // considers such bounds to be unsatisfiable, so
                    // it's kind of a moot point since you could never
                    // construct such an object, but this seems
                    // correct even if that code changes).
                    if t == &erased_self_ty && !r.has_escaping_bound_vars() {
                        Some(*r)
                    } else {
                        None
                    }
                }
            }
        })
        .collect()
}
