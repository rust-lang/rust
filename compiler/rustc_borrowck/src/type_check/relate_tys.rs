use rustc_infer::infer::nll_relate::{NormalizationStrategy, TypeRelating, TypeRelatingDelegate};
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::relate::TypeRelation;
use rustc_middle::ty::{self, Const, Ty};
use rustc_trait_selection::traits::query::Fallible;

use crate::constraints::OutlivesConstraint;
use crate::diagnostics::UniverseInfo;
use crate::type_check::{Locations, TypeChecker};

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    /// Adds sufficient constraints to ensure that `a R b` where `R` depends on `v`:
    ///
    /// - "Covariant" `a <: b`
    /// - "Invariant" `a == b`
    /// - "Contravariant" `a :> b`
    ///
    /// N.B., the type `a` is permitted to have unresolved inference
    /// variables, but not the type `b`.
    #[instrument(skip(self), level = "debug")]
    pub(super) fn relate_types(
        &mut self,
        a: Ty<'tcx>,
        v: ty::Variance,
        b: Ty<'tcx>,
        locations: Locations,
        category: ConstraintCategory,
    ) -> Fallible<()> {
        TypeRelating::new(
            self.infcx,
            NllTypeRelatingDelegate::new(self, locations, category, UniverseInfo::relate(a, b)),
            v,
        )
        .relate(a, b)?;
        Ok(())
    }
}

struct NllTypeRelatingDelegate<'me, 'bccx, 'tcx> {
    type_checker: &'me mut TypeChecker<'bccx, 'tcx>,

    /// Where (and why) is this relation taking place?
    locations: Locations,

    /// What category do we assign the resulting `'a: 'b` relationships?
    category: ConstraintCategory,

    /// Information so that error reporting knows what types we are relating
    /// when reporting a bound region error.
    universe_info: UniverseInfo<'tcx>,
}

impl<'me, 'bccx, 'tcx> NllTypeRelatingDelegate<'me, 'bccx, 'tcx> {
    fn new(
        type_checker: &'me mut TypeChecker<'bccx, 'tcx>,
        locations: Locations,
        category: ConstraintCategory,
        universe_info: UniverseInfo<'tcx>,
    ) -> Self {
        Self { type_checker, locations, category, universe_info }
    }
}

impl<'tcx> TypeRelatingDelegate<'tcx> for NllTypeRelatingDelegate<'_, '_, 'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.type_checker.param_env
    }

    fn create_next_universe(&mut self) -> ty::UniverseIndex {
        let universe = self.type_checker.infcx.create_next_universe();
        self.type_checker
            .borrowck_context
            .constraints
            .universe_causes
            .insert(universe, self.universe_info.clone());
        universe
    }

    fn next_existential_region_var(&mut self, from_forall: bool) -> ty::Region<'tcx> {
        let origin = NllRegionVariableOrigin::Existential { from_forall };
        self.type_checker.infcx.next_nll_region_var(origin)
    }

    fn next_placeholder_region(&mut self, placeholder: ty::PlaceholderRegion) -> ty::Region<'tcx> {
        self.type_checker
            .borrowck_context
            .constraints
            .placeholder_region(self.type_checker.infcx, placeholder)
    }

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        self.type_checker.infcx.next_nll_region_var_in_universe(
            NllRegionVariableOrigin::Existential { from_forall: false },
            universe,
        )
    }

    fn push_outlives(
        &mut self,
        sup: ty::Region<'tcx>,
        sub: ty::Region<'tcx>,
        info: ty::VarianceDiagInfo<'tcx>,
    ) {
        let sub = self.type_checker.borrowck_context.universal_regions.to_region_vid(sub);
        let sup = self.type_checker.borrowck_context.universal_regions.to_region_vid(sup);
        self.type_checker.borrowck_context.constraints.outlives_constraints.push(
            OutlivesConstraint {
                sup,
                sub,
                locations: self.locations,
                category: self.category,
                variance_info: info,
            },
        );
    }

    // We don't have to worry about the equality of consts during borrow checking
    // as consts always have a static lifetime.
    fn const_equate(&mut self, _a: Const<'tcx>, _b: Const<'tcx>) {}

    fn normalization() -> NormalizationStrategy {
        NormalizationStrategy::Eager
    }

    fn forbid_inference_vars() -> bool {
        true
    }
}
