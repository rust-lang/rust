use std::fmt;

use rustc_infer::infer::canonical::Canonical;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{self, ToPredicate, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::traits::query::type_op::{self, TypeOpOutput};
use rustc_trait_selection::traits::query::Fallible;

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
    pub(super) fn fully_perform_op<R, Op>(
        &mut self,
        locations: Locations,
        category: ConstraintCategory,
        op: Op,
    ) -> Fallible<R>
    where
        Op: type_op::TypeOp<'tcx, Output = R>,
        Canonical<'tcx, Op>: ToUniverseInfo<'tcx>,
    {
        let old_universe = self.infcx.universe();

        let TypeOpOutput { output, constraints, canonicalized_query } =
            op.fully_perform(self.infcx)?;

        if let Some(data) = &constraints {
            self.push_region_constraints(locations, category, data);
        }

        let universe = self.infcx.universe();

        if old_universe != universe {
            let universe_info = match canonicalized_query {
                Some(canonicalized_query) => canonicalized_query.to_universe_info(old_universe),
                None => UniverseInfo::other(),
            };
            for u in old_universe..universe {
                let info_universe =
                    self.borrowck_context.constraints.universe_causes.push(universe_info.clone());
                assert_eq!(u.as_u32() + 1, info_universe.as_u32());
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
        let (instantiated, _) =
            self.infcx.instantiate_canonical_with_fresh_inference_vars(span, canonical);

        for _ in 0..canonical.max_universe.as_u32() {
            let info = UniverseInfo::other();
            self.borrowck_context.constraints.universe_causes.push(info);
        }

        instantiated
    }

    pub(super) fn prove_trait_ref(
        &mut self,
        trait_ref: ty::TraitRef<'tcx>,
        locations: Locations,
        category: ConstraintCategory,
    ) {
        self.prove_predicates(
            Some(ty::PredicateKind::Trait(ty::TraitPredicate {
                trait_ref,
                constness: ty::BoundConstness::NotConst,
            })),
            locations,
            category,
        );
    }

    pub(super) fn normalize_and_prove_instantiated_predicates(
        &mut self,
        instantiated_predicates: ty::InstantiatedPredicates<'tcx>,
        locations: Locations,
    ) {
        for predicate in instantiated_predicates.predicates {
            let predicate = self.normalize(predicate, locations);
            self.prove_predicate(predicate, locations, ConstraintCategory::Boring);
        }
    }

    pub(super) fn prove_predicates(
        &mut self,
        predicates: impl IntoIterator<Item = impl ToPredicate<'tcx>>,
        locations: Locations,
        category: ConstraintCategory,
    ) {
        for predicate in predicates {
            let predicate = predicate.to_predicate(self.tcx());
            debug!("prove_predicates(predicate={:?}, locations={:?})", predicate, locations,);

            self.prove_predicate(predicate, locations, category);
        }
    }

    pub(super) fn prove_predicate(
        &mut self,
        predicate: ty::Predicate<'tcx>,
        locations: Locations,
        category: ConstraintCategory,
    ) {
        debug!("prove_predicate(predicate={:?}, location={:?})", predicate, locations,);

        let param_env = self.param_env;
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
        debug!("normalize(value={:?}, location={:?})", value, location);
        let param_env = self.param_env;
        self.fully_perform_op(
            location.to_locations(),
            ConstraintCategory::Boring,
            param_env.and(type_op::normalize::Normalize::new(value)),
        )
        .unwrap_or_else(|NoSolution| {
            span_mirbug!(self, NoSolution, "failed to normalize `{:?}`", value);
            value
        })
    }
}
