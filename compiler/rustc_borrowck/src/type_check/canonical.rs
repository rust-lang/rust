use std::fmt;

use rustc_errors::ErrorGuaranteed;
use rustc_infer::infer::canonical::Canonical;
use rustc_infer::infer::outlives::env::RegionBoundPairs;
use rustc_middle::bug;
use rustc_middle::mir::{Body, ConstraintCategory};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, Upcast};
use rustc_span::Span;
use rustc_span::def_id::DefId;
use rustc_trait_selection::solve::NoSolution;
use rustc_trait_selection::traits::ObligationCause;
use rustc_trait_selection::traits::query::type_op::custom::CustomTypeOp;
use rustc_trait_selection::traits::query::type_op::{self, TypeOpOutput};
use tracing::{debug, instrument};

use super::{Locations, NormalizeLocation, TypeChecker};
use crate::BorrowckInferCtxt;
use crate::diagnostics::ToUniverseInfo;
use crate::type_check::{MirTypeckRegionConstraints, constraint_conversion};
use crate::universal_regions::UniversalRegions;

#[instrument(skip(infcx, constraints, op), level = "trace")]
pub(crate) fn fully_perform_op_raw<'tcx, R: fmt::Debug, Op>(
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &Body<'tcx>,
    universal_regions: &UniversalRegions<'tcx>,
    region_bound_pairs: &RegionBoundPairs<'tcx>,
    known_type_outlives_obligations: &[ty::PolyTypeOutlivesPredicate<'tcx>],
    constraints: &mut MirTypeckRegionConstraints<'tcx>,
    locations: Locations,
    category: ConstraintCategory<'tcx>,
    op: Op,
) -> Result<R, ErrorGuaranteed>
where
    Op: type_op::TypeOp<'tcx, Output = R>,
    Op::ErrorInfo: ToUniverseInfo<'tcx>,
{
    let old_universe = infcx.universe();

    let TypeOpOutput { output, constraints: query_constraints, error_info } =
        op.fully_perform(infcx, infcx.root_def_id, locations.span(body))?;
    if cfg!(debug_assertions) {
        let data = infcx.take_and_reset_region_constraints();
        if !data.is_empty() {
            panic!("leftover region constraints: {data:#?}");
        }
    }

    debug!(?output, ?query_constraints);

    if let Some(data) = query_constraints {
        constraint_conversion::ConstraintConversion::new(
            infcx,
            universal_regions,
            region_bound_pairs,
            known_type_outlives_obligations,
            locations,
            locations.span(body),
            category,
            constraints,
        )
        .convert_all(data);
    }

    // If the query has created new universes and errors are going to be emitted, register the
    // cause of these new universes for improved diagnostics.
    let universe = infcx.universe();
    if old_universe != universe
        && let Some(error_info) = error_info
    {
        let universe_info = error_info.to_universe_info(old_universe);
        for u in (old_universe + 1)..=universe {
            constraints.universe_causes.insert(u, universe_info.clone());
        }
    }

    Ok(output)
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    /// Given some operation `op` that manipulates types, proves
    /// predicates, or otherwise uses the inference context, executes
    /// `op` and then executes all the further obligations that `op`
    /// returns. This will yield a set of outlives constraints amongst
    /// regions which are extracted and stored as having occurred at
    /// `locations`.
    ///
    /// **Any `rustc_infer::infer` operations that might generate region
    /// constraints should occur within this method so that those
    /// constraints can be properly localized!**
    #[instrument(skip(self, op), level = "trace")]
    pub(super) fn fully_perform_op<R: fmt::Debug, Op>(
        &mut self,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
        op: Op,
    ) -> Result<R, ErrorGuaranteed>
    where
        Op: type_op::TypeOp<'tcx, Output = R>,
        Op::ErrorInfo: ToUniverseInfo<'tcx>,
    {
        fully_perform_op_raw(
            self.infcx,
            self.body,
            self.universal_regions,
            self.region_bound_pairs,
            self.known_type_outlives_obligations,
            self.constraints,
            locations,
            category,
            op,
        )
    }

    pub(super) fn instantiate_canonical<T>(
        &mut self,
        span: Span,
        canonical: &Canonical<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        let (instantiated, _) = self.infcx.instantiate_canonical(span, canonical);
        instantiated
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn prove_trait_ref(
        &mut self,
        trait_ref: ty::TraitRef<'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) {
        self.prove_predicate(
            ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::Trait(
                ty::TraitPredicate { trait_ref, polarity: ty::PredicatePolarity::Positive },
            ))),
            locations,
            category,
        );
    }

    #[instrument(level = "debug", skip(self))]
    pub(super) fn normalize_and_prove_instantiated_predicates(
        &mut self,
        // Keep this parameter for now, in case we start using
        // it in `ConstraintCategory` at some point.
        _def_id: DefId,
        instantiated_predicates: ty::InstantiatedPredicates<'tcx>,
        locations: Locations,
    ) {
        for (predicate, span) in instantiated_predicates {
            debug!(?span, ?predicate);
            let category = ConstraintCategory::Predicate(span);
            let predicate = self.normalize_with_category(predicate, locations, category);
            self.prove_predicate(predicate, locations, category);
        }
    }

    pub(super) fn prove_predicates(
        &mut self,
        predicates: impl IntoIterator<Item: Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>> + std::fmt::Debug>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) {
        for predicate in predicates {
            self.prove_predicate(predicate, locations, category);
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn prove_predicate(
        &mut self,
        predicate: impl Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>> + std::fmt::Debug,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) {
        let param_env = self.infcx.param_env;
        let predicate = predicate.upcast(self.tcx());
        let _: Result<_, ErrorGuaranteed> = self.fully_perform_op(
            locations,
            category,
            param_env.and(type_op::prove_predicate::ProvePredicate { predicate }),
        );
    }

    pub(super) fn normalize<T>(&mut self, value: T, location: impl NormalizeLocation) -> T
    where
        T: type_op::normalize::Normalizable<'tcx> + fmt::Display + Copy + 'tcx,
    {
        self.normalize_with_category(value, location, ConstraintCategory::Boring)
    }

    pub(super) fn deeply_normalize<T>(&mut self, value: T, location: impl NormalizeLocation) -> T
    where
        T: type_op::normalize::Normalizable<'tcx> + fmt::Display + Copy + 'tcx,
    {
        let result: Result<_, ErrorGuaranteed> = self.fully_perform_op(
            location.to_locations(),
            ConstraintCategory::Boring,
            self.infcx.param_env.and(type_op::normalize::DeeplyNormalize { value }),
        );
        result.unwrap_or(value)
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn normalize_with_category<T>(
        &mut self,
        value: T,
        location: impl NormalizeLocation,
        category: ConstraintCategory<'tcx>,
    ) -> T
    where
        T: type_op::normalize::Normalizable<'tcx> + fmt::Display + Copy + 'tcx,
    {
        let param_env = self.infcx.param_env;
        let result: Result<_, ErrorGuaranteed> = self.fully_perform_op(
            location.to_locations(),
            category,
            param_env.and(type_op::normalize::Normalize { value }),
        );
        result.unwrap_or(value)
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn struct_tail(
        &mut self,
        ty: Ty<'tcx>,
        location: impl NormalizeLocation,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();
        let body = self.body;

        let cause = ObligationCause::misc(
            location.to_locations().span(body),
            body.source.def_id().expect_local(),
        );

        if self.infcx.next_trait_solver() {
            let param_env = self.infcx.param_env;
            // FIXME: Make this into a real type op?
            self.fully_perform_op(
                location.to_locations(),
                ConstraintCategory::Boring,
                CustomTypeOp::new(
                    |ocx| {
                        let structurally_normalize = |ty| {
                            ocx.structurally_normalize_ty(
                                &cause,
                                param_env,
                                ty,
                            )
                            .unwrap_or_else(|_| bug!("struct tail should have been computable, since we computed it in HIR"))
                        };

                        let tail = tcx.struct_tail_raw(
                            ty,
                            &cause,
                            structurally_normalize,
                            || {},
                        );

                        Ok(tail)
                    },
                    "normalizing struct tail",
                ),
            )
            .unwrap_or_else(|guar| Ty::new_error(tcx, guar))
        } else {
            let mut normalize = |ty| self.normalize(ty, location);
            let tail = tcx.struct_tail_raw(ty, &cause, &mut normalize, || {});
            normalize(tail)
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn structurally_resolve(
        &mut self,
        ty: Ty<'tcx>,
        location: impl NormalizeLocation,
    ) -> Ty<'tcx> {
        if self.infcx.next_trait_solver() {
            let body = self.body;
            let param_env = self.infcx.param_env;
            // FIXME: Make this into a real type op?
            self.fully_perform_op(
                location.to_locations(),
                ConstraintCategory::Boring,
                CustomTypeOp::new(
                    |ocx| {
                        ocx.structurally_normalize_ty(
                            &ObligationCause::misc(
                                location.to_locations().span(body),
                                body.source.def_id().expect_local(),
                            ),
                            param_env,
                            ty,
                        )
                        .map_err(|_| NoSolution)
                    },
                    "normalizing struct tail",
                ),
            )
            .unwrap_or_else(|guar| Ty::new_error(self.tcx(), guar))
        } else {
            self.normalize(ty, location)
        }
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn ascribe_user_type(
        &mut self,
        mir_ty: Ty<'tcx>,
        user_ty: ty::UserType<'tcx>,
        span: Span,
    ) {
        let _: Result<_, ErrorGuaranteed> = self.fully_perform_op(
            Locations::All(span),
            ConstraintCategory::Boring,
            self.infcx
                .param_env
                .and(type_op::ascribe_user_type::AscribeUserType { mir_ty, user_ty }),
        );
    }

    /// *Incorrectly* skips the WF checks we normally do in `ascribe_user_type`.
    ///
    /// FIXME(#104478, #104477): This is a hack for backward-compatibility.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn ascribe_user_type_skip_wf(
        &mut self,
        mir_ty: Ty<'tcx>,
        user_ty: ty::UserType<'tcx>,
        span: Span,
    ) {
        let ty::UserTypeKind::Ty(user_ty) = user_ty.kind else { bug!() };

        // A fast path for a common case with closure input/output types.
        if let ty::Infer(_) = user_ty.kind() {
            self.eq_types(user_ty, mir_ty, Locations::All(span), ConstraintCategory::Boring)
                .unwrap();
            return;
        }

        // FIXME: Ideally MIR types are normalized, but this is not always true.
        let mir_ty = self.normalize(mir_ty, Locations::All(span));

        let cause = ObligationCause::dummy_with_span(span);
        let param_env = self.infcx.param_env;
        let _: Result<_, ErrorGuaranteed> = self.fully_perform_op(
            Locations::All(span),
            ConstraintCategory::Boring,
            type_op::custom::CustomTypeOp::new(
                |ocx| {
                    let user_ty = ocx.normalize(&cause, param_env, user_ty);
                    ocx.eq(&cause, param_env, user_ty, mir_ty)?;
                    Ok(())
                },
                "ascribe_user_type_skip_wf",
            ),
        );
    }
}
