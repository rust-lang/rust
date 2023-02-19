use std::fmt;

use rustc_infer::infer::{canonical::Canonical, InferOk};
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{self, ToPredicate, Ty, TypeFoldable};
use rustc_span::def_id::DefId;
use rustc_span::Span;
use rustc_trait_selection::traits::query::type_op::{self, TypeOpOutput};
use rustc_trait_selection::traits::query::{Fallible, NoSolution};
use rustc_trait_selection::traits::{ObligationCause, ObligationCtxt};

use crate::diagnostics::{ToUniverseInfo, UniverseInfo};

use super::{Locations, NormalizeLocation, TypeChecker};

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
    ) -> Fallible<R>
    where
        Op: type_op::TypeOp<'tcx, Output = R>,
        Op::ErrorInfo: ToUniverseInfo<'tcx>,
    {
        let old_universe = self.infcx.universe();

        let TypeOpOutput { output, constraints, error_info } = op.fully_perform(self.infcx)?;

        debug!(?output, ?constraints);

        if let Some(data) = constraints {
            self.push_region_constraints(locations, category, data);
        }

        let universe = self.infcx.universe();

        if old_universe != universe {
            let universe_info = match error_info {
                Some(error_info) => error_info.to_universe_info(old_universe),
                None => UniverseInfo::other(),
            };
            for u in (old_universe + 1)..=universe {
                self.borrowck_context.constraints.universe_causes.insert(u, universe_info.clone());
            }
        }

        Ok(output)
    }

    pub(super) fn instantiate_canonical_with_fresh_inference_vars<T>(
        &mut self,
        span: Span,
        canonical: &Canonical<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        let old_universe = self.infcx.universe();

        let (instantiated, _) =
            self.infcx.instantiate_canonical_with_fresh_inference_vars(span, canonical);

        for u in (old_universe + 1)..=self.infcx.universe() {
            self.borrowck_context.constraints.universe_causes.insert(u, UniverseInfo::other());
        }

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
            ty::Binder::dummy(ty::PredicateKind::Clause(ty::Clause::Trait(ty::TraitPredicate {
                trait_ref,
                constness: ty::BoundConstness::NotConst,
                polarity: ty::ImplPolarity::Positive,
            }))),
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
            debug!(?predicate);
            let category = ConstraintCategory::Predicate(span);
            let predicate = self.normalize_with_category(predicate, locations, category);
            self.prove_predicate(predicate, locations, category);
        }
    }

    pub(super) fn prove_predicates(
        &mut self,
        predicates: impl IntoIterator<Item = impl ToPredicate<'tcx> + std::fmt::Debug>,
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
        predicate: impl ToPredicate<'tcx> + std::fmt::Debug,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) {
        let param_env = self.param_env;
        let predicate = predicate.to_predicate(self.tcx());
        self.fully_perform_op(
            locations,
            category,
            param_env.and(type_op::prove_predicate::ProvePredicate::new(predicate)),
        )
        .unwrap_or_else(|NoSolution| {
            span_mirbug!(self, NoSolution, "could not prove {:?}", predicate);
        })
    }

    pub(super) fn normalize<T>(&mut self, value: T, location: impl NormalizeLocation) -> T
    where
        T: type_op::normalize::Normalizable<'tcx> + fmt::Display + Copy + 'tcx,
    {
        self.normalize_with_category(value, location, ConstraintCategory::Boring)
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
        let param_env = self.param_env;
        self.fully_perform_op(
            location.to_locations(),
            category,
            param_env.and(type_op::normalize::Normalize::new(value)),
        )
        .unwrap_or_else(|NoSolution| {
            span_mirbug!(self, NoSolution, "failed to normalize `{:?}`", value);
            value
        })
    }

    #[instrument(skip(self), level = "debug")]
    pub(super) fn ascribe_user_type(
        &mut self,
        mir_ty: Ty<'tcx>,
        user_ty: ty::UserType<'tcx>,
        span: Span,
    ) {
        self.fully_perform_op(
            Locations::All(span),
            ConstraintCategory::Boring,
            self.param_env.and(type_op::ascribe_user_type::AscribeUserType::new(mir_ty, user_ty)),
        )
        .unwrap_or_else(|err| {
            span_mirbug!(
                self,
                span,
                "ascribe_user_type `{mir_ty:?}=={user_ty:?}` failed with `{err:?}`",
            );
        });
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
        let ty::UserType::Ty(user_ty) = user_ty else { bug!() };

        // A fast path for a common case with closure input/output types.
        if let ty::Infer(_) = user_ty.kind() {
            self.eq_types(user_ty, mir_ty, Locations::All(span), ConstraintCategory::Boring)
                .unwrap();
            return;
        }

        // FIXME: Ideally MIR types are normalized, but this is not always true.
        let mir_ty = self.normalize(mir_ty, Locations::All(span));

        let cause = ObligationCause::dummy_with_span(span);
        let param_env = self.param_env;
        let op = |infcx: &'_ _| {
            let ocx = ObligationCtxt::new_in_snapshot(infcx);
            let user_ty = ocx.normalize(&cause, param_env, user_ty);
            ocx.eq(&cause, param_env, user_ty, mir_ty)?;
            if !ocx.select_all_or_error().is_empty() {
                return Err(NoSolution);
            }
            Ok(InferOk { value: (), obligations: vec![] })
        };

        self.fully_perform_op(
            Locations::All(span),
            ConstraintCategory::Boring,
            type_op::custom::CustomTypeOp::new(op, || "ascribe_user_type_skip_wf".to_string()),
        )
        .unwrap_or_else(|err| {
            span_mirbug!(
                self,
                span,
                "ascribe_user_type_skip_wf `{mir_ty:?}=={user_ty:?}` failed with `{err:?}`",
            );
        });
    }
}
