//! Core logic responsible for determining what it means for various type system
//! primitives to be "well formed". Actually checking whether these primitives are
//! well formed is performed elsewhere (e.g. during type checking or item well formedness
//! checking).

use std::iter;

use rustc_hir as hir;
use rustc_hir::lang_items::LangItem;
use rustc_infer::traits::{ObligationCauseCode, PredicateObligations};
use rustc_middle::bug;
use rustc_middle::ty::{
    self, GenericArg, GenericArgKind, GenericArgsRef, Ty, TyCtxt, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor,
};
use rustc_session::parse::feature_err;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{Span, sym};
use tracing::{debug, instrument, trace};

use crate::infer::InferCtxt;
use crate::traits;

/// Returns the set of obligations needed to make `arg` well-formed.
/// If `arg` contains unresolved inference variables, this may include
/// further WF obligations. However, if `arg` IS an unresolved
/// inference variable, returns `None`, because we are not able to
/// make any progress at all. This is to prevent cycles where we
/// say "?0 is WF if ?0 is WF".
pub fn obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    recursion_depth: usize,
    arg: GenericArg<'tcx>,
    span: Span,
) -> Option<PredicateObligations<'tcx>> {
    // Handle the "cycle" case (see comment above) by bailing out if necessary.
    let arg = match arg.unpack() {
        GenericArgKind::Type(ty) => {
            match ty.kind() {
                ty::Infer(ty::TyVar(_)) => {
                    let resolved_ty = infcx.shallow_resolve(ty);
                    if resolved_ty == ty {
                        // No progress, bail out to prevent cycles.
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
                    let resolved = infcx.shallow_resolve_const(ct);
                    if resolved == ct {
                        // No progress, bail out to prevent cycles.
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
        GenericArgKind::Lifetime(..) => return Some(PredicateObligations::new()),
    };

    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id,
        span,
        out: PredicateObligations::new(),
        recursion_depth,
        item: None,
    };
    wf.add_wf_preds_for_generic_arg(arg);
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
    span: Span,
    body_id: LocalDefId,
) -> Option<PredicateObligations<'tcx>> {
    debug_assert_eq!(arg, infcx.resolve_vars_if_possible(arg));

    // However, if `arg` IS an unresolved inference variable, returns `None`,
    // because we are not able to make any progress at all. This is to prevent
    // cycles where we say "?0 is WF if ?0 is WF".
    if arg.is_non_region_infer() {
        return None;
    }

    if let ty::GenericArgKind::Lifetime(..) = arg.unpack() {
        return Some(PredicateObligations::new());
    }

    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id,
        span,
        out: PredicateObligations::new(),
        recursion_depth: 0,
        item: None,
    };
    wf.add_wf_preds_for_generic_arg(arg);
    Some(wf.out)
}

/// Returns the obligations that make this trait reference
/// well-formed. For example, if there is a trait `Set` defined like
/// `trait Set<K: Eq>`, then the trait bound `Foo: Set<Bar>` is WF
/// if `Bar: Eq`.
pub fn trait_obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    trait_pred: ty::TraitPredicate<'tcx>,
    span: Span,
    item: &'tcx hir::Item<'tcx>,
) -> PredicateObligations<'tcx> {
    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id,
        span,
        out: PredicateObligations::new(),
        recursion_depth: 0,
        item: Some(item),
    };
    wf.add_wf_preds_for_trait_pred(trait_pred, Elaborate::All);
    debug!(obligations = ?wf.out);
    wf.normalize(infcx)
}

/// Returns the requirements for `clause` to be well-formed.
///
/// For example, if there is a trait `Set` defined like
/// `trait Set<K: Eq>`, then the trait bound `Foo: Set<Bar>` is WF
/// if `Bar: Eq`.
#[instrument(skip(infcx), ret)]
pub fn clause_obligations<'tcx>(
    infcx: &InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    clause: ty::Clause<'tcx>,
    span: Span,
) -> PredicateObligations<'tcx> {
    let mut wf = WfPredicates {
        infcx,
        param_env,
        body_id,
        span,
        out: PredicateObligations::new(),
        recursion_depth: 0,
        item: None,
    };

    // It's ok to skip the binder here because wf code is prepared for it
    match clause.kind().skip_binder() {
        ty::ClauseKind::Trait(t) => {
            wf.add_wf_preds_for_trait_pred(t, Elaborate::None);
        }
        ty::ClauseKind::HostEffect(..) => {
            // Technically the well-formedness of this predicate is implied by
            // the corresponding trait predicate it should've been generated beside.
        }
        ty::ClauseKind::RegionOutlives(..) => {}
        ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ty, _reg)) => {
            wf.add_wf_preds_for_generic_arg(ty.into());
        }
        ty::ClauseKind::Projection(t) => {
            wf.add_wf_preds_for_alias_term(t.projection_term);
            wf.add_wf_preds_for_generic_arg(t.term.into_arg());
        }
        ty::ClauseKind::ConstArgHasType(ct, ty) => {
            wf.add_wf_preds_for_generic_arg(ct.into());
            wf.add_wf_preds_for_generic_arg(ty.into());
        }
        ty::ClauseKind::WellFormed(arg) => {
            wf.add_wf_preds_for_generic_arg(arg);
        }

        ty::ClauseKind::ConstEvaluatable(ct) => {
            wf.add_wf_preds_for_generic_arg(ct.into());
        }
    }

    wf.normalize(infcx)
}

struct WfPredicates<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    span: Span,
    out: PredicateObligations<'tcx>,
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

/// Points the cause span of a super predicate at the relevant associated type.
///
/// Given a trait impl item:
///
/// ```ignore (incomplete)
/// impl TargetTrait for TargetType {
///    type Assoc = SomeType;
/// }
/// ```
///
/// And a super predicate of `TargetTrait` that has any of the following forms:
///
/// 1. `<OtherType as OtherTrait>::Assoc == <TargetType as TargetTrait>::Assoc`
/// 2. `<<TargetType as TargetTrait>::Assoc as OtherTrait>::Assoc == OtherType`
/// 3. `<TargetType as TargetTrait>::Assoc: OtherTrait`
///
/// Replace the span of the cause with the span of the associated item:
///
/// ```ignore (incomplete)
/// impl TargetTrait for TargetType {
///     type Assoc = SomeType;
/// //               ^^^^^^^^ this span
/// }
/// ```
///
/// Note that bounds that can be expressed as associated item bounds are **not**
/// super predicates. This means that form 2 and 3 from above are only relevant if
/// the [`GenericArgsRef`] of the projection type are not its identity arguments.
fn extend_cause_with_original_assoc_item_obligation<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: Option<&hir::Item<'tcx>>,
    cause: &mut traits::ObligationCause<'tcx>,
    pred: ty::Predicate<'tcx>,
) {
    debug!(?item, ?cause, ?pred, "extended_cause_with_original_assoc_item_obligation");
    let (items, impl_def_id) = match item {
        Some(hir::Item { kind: hir::ItemKind::Impl(impl_), owner_id, .. }) => {
            (impl_.items, *owner_id)
        }
        _ => return,
    };

    let ty_to_impl_span = |ty: Ty<'_>| {
        if let ty::Alias(ty::Projection, projection_ty) = ty.kind()
            && let Some(&impl_item_id) =
                tcx.impl_item_implementor_ids(impl_def_id).get(&projection_ty.def_id)
            && let Some(impl_item) =
                items.iter().find(|item| item.id.owner_id.to_def_id() == impl_item_id)
        {
            Some(tcx.hir_impl_item(impl_item.id).expect_type().span)
        } else {
            None
        }
    };

    // It is fine to skip the binder as we don't care about regions here.
    match pred.kind().skip_binder() {
        ty::PredicateKind::Clause(ty::ClauseKind::Projection(proj)) => {
            // Form 1: The obligation comes not from the current `impl` nor the `trait` being
            // implemented, but rather from a "second order" obligation, where an associated
            // type has a projection coming from another associated type.
            // See `tests/ui/traits/assoc-type-in-superbad.rs` for an example.
            if let Some(term_ty) = proj.term.as_type()
                && let Some(impl_item_span) = ty_to_impl_span(term_ty)
            {
                cause.span = impl_item_span;
            }

            // Form 2: A projection obligation for an associated item failed to be met.
            // We overwrite the span from above to ensure that a bound like
            // `Self::Assoc1: Trait<OtherAssoc = Self::Assoc2>` gets the same
            // span for both obligations that it is lowered to.
            if let Some(impl_item_span) = ty_to_impl_span(proj.self_ty()) {
                cause.span = impl_item_span;
            }
        }

        ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
            // Form 3: A trait obligation for an associated item failed to be met.
            debug!("extended_cause_with_original_assoc_item_obligation trait proj {:?}", pred);
            if let Some(impl_item_span) = ty_to_impl_span(pred.self_ty()) {
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

    fn normalize(self, infcx: &InferCtxt<'tcx>) -> PredicateObligations<'tcx> {
        // Do not normalize `wf` obligations with the new solver.
        //
        // The current deep normalization routine with the new solver does not
        // handle ambiguity and the new solver correctly deals with unnnormalized goals.
        // If the user relies on normalized types, e.g. for `fn implied_outlives_bounds`,
        // it is their responsibility to normalize while avoiding ambiguity.
        if infcx.next_trait_solver() {
            return self.out;
        }

        let cause = self.cause(ObligationCauseCode::WellFormed(None));
        let param_env = self.param_env;
        let mut obligations = PredicateObligations::with_capacity(self.out.len());
        for mut obligation in self.out {
            assert!(!obligation.has_escaping_bound_vars());
            let mut selcx = traits::SelectionContext::new(infcx);
            // Don't normalize the whole obligation, the param env is either
            // already normalized, or we're currently normalizing the
            // param_env. Either way we should only normalize the predicate.
            let normalized_predicate = traits::normalize::normalize_with_depth_to(
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
    fn add_wf_preds_for_trait_pred(
        &mut self,
        trait_pred: ty::TraitPredicate<'tcx>,
        elaborate: Elaborate,
    ) {
        let tcx = self.tcx();
        let trait_ref = trait_pred.trait_ref;

        // Negative trait predicates don't require supertraits to hold, just
        // that their args are WF.
        if trait_pred.polarity == ty::PredicatePolarity::Negative {
            self.add_wf_preds_for_negative_trait_pred(trait_ref);
            return;
        }

        // if the trait predicate is not const, the wf obligations should not be const as well.
        let obligations = self.nominal_obligations(trait_ref.def_id, trait_ref.args);

        debug!("compute_trait_pred obligations {:?}", obligations);
        let param_env = self.param_env;
        let depth = self.recursion_depth;

        let item = self.item;

        let extend = |traits::PredicateObligation { predicate, mut cause, .. }| {
            if let Some(parent_trait_pred) = predicate.as_trait_clause() {
                cause = cause.derived_cause(
                    parent_trait_pred,
                    traits::ObligationCauseCode::WellFormedDerived,
                );
            }
            extend_cause_with_original_assoc_item_obligation(tcx, item, &mut cause, predicate);
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
                .args
                .iter()
                .enumerate()
                .filter(|(_, arg)| {
                    matches!(arg.unpack(), GenericArgKind::Type(..) | GenericArgKind::Const(..))
                })
                .filter(|(_, arg)| !arg.has_escaping_bound_vars())
                .map(|(i, arg)| {
                    let mut cause = traits::ObligationCause::misc(self.span, self.body_id);
                    // The first arg is the self ty - use the correct span for it.
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
    fn add_wf_preds_for_negative_trait_pred(&mut self, trait_ref: ty::TraitRef<'tcx>) {
        for arg in trait_ref.args {
            self.add_wf_preds_for_generic_arg(arg);
        }
    }

    /// Pushes the obligations required for an alias (except inherent) to be WF
    /// into `self.out`.
    fn add_wf_preds_for_alias_term(&mut self, data: ty::AliasTerm<'tcx>) {
        // A projection is well-formed if
        //
        // (a) its predicates hold (*)
        // (b) its args are wf
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
        let obligations = self.nominal_obligations(data.def_id, data.args);
        self.out.extend(obligations);

        self.add_wf_preds_for_projection_args(data.args);
    }

    /// Pushes the obligations required for an inherent alias to be WF
    /// into `self.out`.
    // FIXME(inherent_associated_types): Merge this function with `fn compute_alias`.
    fn add_wf_preds_for_inherent_projection(&mut self, data: ty::AliasTy<'tcx>) {
        // An inherent projection is well-formed if
        //
        // (a) its predicates hold (*)
        // (b) its args are wf
        //
        // (*) The predicates of an inherent associated type include the
        //     predicates of the impl that it's contained in.

        if !data.self_ty().has_escaping_bound_vars() {
            // FIXME(inherent_associated_types): Should this happen inside of a snapshot?
            // FIXME(inherent_associated_types): This is incompatible with the new solver and lazy norm!
            let args = traits::project::compute_inherent_assoc_ty_args(
                &mut traits::SelectionContext::new(self.infcx),
                self.param_env,
                data,
                self.cause(ObligationCauseCode::WellFormed(None)),
                self.recursion_depth,
                &mut self.out,
            );
            let obligations = self.nominal_obligations(data.def_id, args);
            self.out.extend(obligations);
        }

        data.args.visit_with(self);
    }

    fn add_wf_preds_for_projection_args(&mut self, args: GenericArgsRef<'tcx>) {
        let tcx = self.tcx();
        let cause = self.cause(ObligationCauseCode::WellFormed(None));
        let param_env = self.param_env;
        let depth = self.recursion_depth;

        self.out.extend(
            args.iter()
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
            let trait_ref = ty::TraitRef::new(
                self.tcx(),
                self.tcx().require_lang_item(LangItem::Sized, Some(cause.span)),
                [subty],
            );
            self.out.push(traits::Obligation::with_depth(
                self.tcx(),
                cause,
                self.recursion_depth,
                self.param_env,
                ty::Binder::dummy(trait_ref),
            ));
        }
    }

    /// Pushes all the predicates needed to validate that `ty` is WF into `out`.
    #[instrument(level = "debug", skip(self))]
    fn add_wf_preds_for_generic_arg(&mut self, arg: GenericArg<'tcx>) {
        arg.visit_with(self);
        debug!(?self.out);
    }

    #[instrument(level = "debug", skip(self))]
    fn nominal_obligations(
        &mut self,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> PredicateObligations<'tcx> {
        let predicates = self.tcx().predicates_of(def_id);
        let mut origins = vec![def_id; predicates.predicates.len()];
        let mut head = predicates;
        while let Some(parent) = head.parent {
            head = self.tcx().predicates_of(parent);
            origins.extend(iter::repeat(parent).take(head.predicates.len()));
        }

        let predicates = predicates.instantiate(self.tcx(), args);
        trace!("{:#?}", predicates);
        debug_assert_eq!(predicates.predicates.len(), origins.len());

        iter::zip(predicates, origins.into_iter().rev())
            .map(|((pred, span), origin_def_id)| {
                let code = ObligationCauseCode::WhereClause(origin_def_id, span);
                let cause = self.cause(code);
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

    fn add_wf_preds_for_dyn_ty(
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
                let cause = self.cause(ObligationCauseCode::ObjectTypeBound(ty, explicit_bound));
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

            // We don't add any wf predicates corresponding to the trait ref's generic arguments
            // which allows code like this to compile:
            // ```rust
            // trait Trait<T: Sized> {}
            // fn foo(_: &dyn Trait<[u32]>) {}
            // ```
        }
    }
}

impl<'a, 'tcx> TypeVisitor<TyCtxt<'tcx>> for WfPredicates<'a, 'tcx> {
    fn visit_ty(&mut self, t: Ty<'tcx>) -> Self::Result {
        debug!("wf bounds for t={:?} t.kind={:#?}", t, t.kind());

        let tcx = self.tcx();

        match *t.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Error(_)
            | ty::Str
            | ty::CoroutineWitness(..)
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
                self.require_sized(subty, ObligationCauseCode::SliceOrArrayElem);
            }

            ty::Array(subty, len) => {
                self.require_sized(subty, ObligationCauseCode::SliceOrArrayElem);
                // Note that the len being WF is implicitly checked while visiting.
                // Here we just check that it's of type usize.
                let cause = self.cause(ObligationCauseCode::ArrayLen(t));
                self.out.push(traits::Obligation::with_depth(
                    tcx,
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::ConstArgHasType(
                        len,
                        tcx.types.usize,
                    ))),
                ));
            }

            ty::Pat(subty, pat) => {
                self.require_sized(subty, ObligationCauseCode::Misc);
                match *pat {
                    ty::PatternKind::Range { start, end } => {
                        let mut check = |c| {
                            let cause = self.cause(ObligationCauseCode::Misc);
                            self.out.push(traits::Obligation::with_depth(
                                tcx,
                                cause.clone(),
                                self.recursion_depth,
                                self.param_env,
                                ty::Binder::dummy(ty::PredicateKind::Clause(
                                    ty::ClauseKind::ConstArgHasType(c, subty),
                                )),
                            ));
                            if !tcx.features().generic_pattern_types() {
                                if c.has_param() {
                                    if self.span.is_dummy() {
                                        self.tcx().dcx().delayed_bug(
                                            "feature error should be reported elsewhere, too",
                                        );
                                    } else {
                                        feature_err(
                                            &self.tcx().sess,
                                            sym::generic_pattern_types,
                                            self.span,
                                            "wraparound pattern type ranges cause monomorphization time errors",
                                        )
                                        .emit();
                                    }
                                }
                            }
                        };
                        check(start);
                        check(end);
                    }
                }
            }

            ty::Tuple(tys) => {
                if let Some((_last, rest)) = tys.split_last() {
                    for &elem in rest {
                        self.require_sized(elem, ObligationCauseCode::TupleElem);
                    }
                }
            }

            ty::RawPtr(_, _) => {
                // Simple cases that are WF if their type args are WF.
            }

            ty::Alias(ty::Projection | ty::Opaque | ty::Weak, data) => {
                let obligations = self.nominal_obligations(data.def_id, data.args);
                self.out.extend(obligations);
            }
            ty::Alias(ty::Inherent, data) => {
                self.add_wf_preds_for_inherent_projection(data);
                return; // Subtree handled by compute_inherent_projection.
            }

            ty::Adt(def, args) => {
                // WfNominalType
                let obligations = self.nominal_obligations(def.did(), args);
                self.out.extend(obligations);
            }

            ty::FnDef(did, args) => {
                // HACK: Check the return type of function definitions for
                // well-formedness to mostly fix #84533. This is still not
                // perfect and there may be ways to abuse the fact that we
                // ignore requirements with escaping bound vars. That's a
                // more general issue however.
                let fn_sig = tcx.fn_sig(did).instantiate(tcx, args);
                fn_sig.output().skip_binder().visit_with(self);

                let obligations = self.nominal_obligations(did, args);
                self.out.extend(obligations);
            }

            ty::Ref(r, rty, _) => {
                // WfReference
                if !r.has_escaping_bound_vars() && !rty.has_escaping_bound_vars() {
                    let cause = self.cause(ObligationCauseCode::ReferenceOutlivesReferent(t));
                    self.out.push(traits::Obligation::with_depth(
                        tcx,
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(
                            ty::OutlivesPredicate(rty, r),
                        ))),
                    ));
                }
            }

            ty::Coroutine(did, args, ..) => {
                // Walk ALL the types in the coroutine: this will
                // include the upvar types as well as the yield
                // type. Note that this is mildly distinct from
                // the closure case, where we have to be careful
                // about the signature of the closure. We don't
                // have the problem of implied bounds here since
                // coroutines don't take arguments.
                let obligations = self.nominal_obligations(did, args);
                self.out.extend(obligations);
            }

            ty::Closure(did, args) => {
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
                let obligations = self.nominal_obligations(did, args);
                self.out.extend(obligations);
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
                let upvars = args.as_closure().tupled_upvars_ty();
                return upvars.visit_with(self);
            }

            ty::CoroutineClosure(did, args) => {
                // See the above comments. The same apply to coroutine-closures.
                let obligations = self.nominal_obligations(did, args);
                self.out.extend(obligations);
                let upvars = args.as_coroutine_closure().tupled_upvars_ty();
                return upvars.visit_with(self);
            }

            ty::FnPtr(..) => {
                // Let the visitor iterate into the argument/return
                // types appearing in the fn signature.
            }
            ty::UnsafeBinder(ty) => {
                // FIXME(unsafe_binders): For now, we have no way to express
                // that a type must be `ManuallyDrop` OR `Copy` (or a pointer).
                if !ty.has_escaping_bound_vars() {
                    self.out.push(traits::Obligation::new(
                        self.tcx(),
                        self.cause(ObligationCauseCode::Misc),
                        self.param_env,
                        ty.map_bound(|ty| {
                            ty::TraitRef::new(
                                self.tcx(),
                                self.tcx().require_lang_item(
                                    LangItem::BikeshedGuaranteedNoDrop,
                                    Some(self.span),
                                ),
                                [ty],
                            )
                        }),
                    ));
                }

                // We recurse into the binder below.
            }

            ty::Dynamic(data, r, _) => {
                // WfObject
                //
                // Here, we defer WF checking due to higher-ranked
                // regions. This is perhaps not ideal.
                self.add_wf_preds_for_dyn_ty(t, data, r);

                // FIXME(#27579) RFC also considers adding trait
                // obligations that don't refer to Self and
                // checking those
                if let Some(principal) = data.principal_def_id() {
                    self.out.push(traits::Obligation::with_depth(
                        tcx,
                        self.cause(ObligationCauseCode::WellFormed(None)),
                        self.recursion_depth,
                        self.param_env,
                        ty::Binder::dummy(ty::PredicateKind::DynCompatible(principal)),
                    ));
                }
            }

            // Inference variables are the complicated case, since we don't
            // know what type they are. We do two things:
            //
            // 1. Check if they have been resolved, and if so proceed with
            //    THAT type.
            // 2. If not, we've at least simplified things (e.g., we went
            //    from `Vec?0>: WF` to `?0: WF`), so we can
            //    register a pending obligation and keep
            //    moving. (Goal is that an "inductive hypothesis"
            //    is satisfied to ensure termination.)
            // See also the comment on `fn obligations`, describing cycle
            // prevention, which happens before this can be reached.
            ty::Infer(_) => {
                let cause = self.cause(ObligationCauseCode::WellFormed(None));
                self.out.push(traits::Obligation::with_depth(
                    tcx,
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                        t.into(),
                    ))),
                ));
            }
        }

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: ty::Const<'tcx>) -> Self::Result {
        let tcx = self.tcx();

        match c.kind() {
            ty::ConstKind::Unevaluated(uv) => {
                if !c.has_escaping_bound_vars() {
                    let obligations = self.nominal_obligations(uv.def, uv.args);
                    self.out.extend(obligations);

                    let predicate = ty::Binder::dummy(ty::PredicateKind::Clause(
                        ty::ClauseKind::ConstEvaluatable(c),
                    ));
                    let cause = self.cause(ObligationCauseCode::WellFormed(None));
                    self.out.push(traits::Obligation::with_depth(
                        tcx,
                        cause,
                        self.recursion_depth,
                        self.param_env,
                        predicate,
                    ));
                }
            }
            ty::ConstKind::Infer(_) => {
                let cause = self.cause(ObligationCauseCode::WellFormed(None));

                self.out.push(traits::Obligation::with_depth(
                    tcx,
                    cause,
                    self.recursion_depth,
                    self.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                        c.into(),
                    ))),
                ));
            }
            ty::ConstKind::Expr(_) => {
                // FIXME(generic_const_exprs): this doesn't verify that given `Expr(N + 1)` the
                // trait bound `typeof(N): Add<typeof(1)>` holds. This is currently unnecessary
                // as `ConstKind::Expr` is only produced via normalization of `ConstKind::Unevaluated`
                // which means that the `DefId` would have been typeck'd elsewhere. However in
                // the future we may allow directly lowering to `ConstKind::Expr` in which case
                // we would not be proving bounds we should.

                let predicate = ty::Binder::dummy(ty::PredicateKind::Clause(
                    ty::ClauseKind::ConstEvaluatable(c),
                ));
                let cause = self.cause(ObligationCauseCode::WellFormed(None));
                self.out.push(traits::Obligation::with_depth(
                    tcx,
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

        c.super_visit_with(self)
    }

    fn visit_predicate(&mut self, _p: ty::Predicate<'tcx>) -> Self::Result {
        bug!("predicate should not be checked for well-formedness");
    }
}

/// Given an object type like `SomeTrait + Send`, computes the lifetime
/// bounds that must hold on the elided self type. These are derived
/// from the declarations of `SomeTrait`, `Send`, and friends -- if
/// they declare `trait SomeTrait : 'static`, for example, then
/// `'static` would appear in the list.
///
/// N.B., in some cases, particularly around higher-ranked bounds,
/// this function returns a kind of conservative approximation.
/// That is, all regions returned by this function are definitely
/// required, but there may be other region bounds that are not
/// returned, as well as requirements like `for<'a> T: 'a`.
///
/// Requires that trait definitions have been processed so that we can
/// elaborate predicates and walk supertraits.
pub fn object_region_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    existential_predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
) -> Vec<ty::Region<'tcx>> {
    let erased_self_ty = tcx.types.trait_object_dummy_self;

    let predicates =
        existential_predicates.iter().map(|predicate| predicate.with_self_ty(tcx, erased_self_ty));

    traits::elaborate(tcx, predicates)
        .filter_map(|pred| {
            debug!(?pred);
            match pred.kind().skip_binder() {
                ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(ref t, ref r)) => {
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
                ty::ClauseKind::Trait(_)
                | ty::ClauseKind::HostEffect(..)
                | ty::ClauseKind::RegionOutlives(_)
                | ty::ClauseKind::Projection(_)
                | ty::ClauseKind::ConstArgHasType(_, _)
                | ty::ClauseKind::WellFormed(_)
                | ty::ClauseKind::ConstEvaluatable(_) => None,
            }
        })
        .collect()
}
