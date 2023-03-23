use rustc_infer::infer::nll_relate::{TypeRelating, TypeRelatingDelegate};
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_infer::traits::PredicateObligations;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::relate::TypeRelation;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, Symbol};
use rustc_trait_selection::traits::query::Fallible;

use crate::constraints::OutlivesConstraint;
use crate::diagnostics::UniverseInfo;
use crate::renumber::{BoundRegionInfo, RegionCtxt};
use crate::type_check::{InstantiateOpaqueType, Locations, TypeChecker};

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
        category: ConstraintCategory<'tcx>,
    ) -> Fallible<()> {
        TypeRelating::new(
            self.infcx,
            NllTypeRelatingDelegate::new(self, locations, category, UniverseInfo::relate(a, b)),
            v,
        )
        .relate(a, b)?;
        Ok(())
    }

    /// Add sufficient constraints to ensure `a == b`. See also [Self::relate_types].
    pub(super) fn eq_substs(
        &mut self,
        a: ty::SubstsRef<'tcx>,
        b: ty::SubstsRef<'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Fallible<()> {
        TypeRelating::new(
            self.infcx,
            NllTypeRelatingDelegate::new(self, locations, category, UniverseInfo::other()),
            ty::Variance::Invariant,
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
    category: ConstraintCategory<'tcx>,

    /// Information so that error reporting knows what types we are relating
    /// when reporting a bound region error.
    universe_info: UniverseInfo<'tcx>,
}

impl<'me, 'bccx, 'tcx> NllTypeRelatingDelegate<'me, 'bccx, 'tcx> {
    fn new(
        type_checker: &'me mut TypeChecker<'bccx, 'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
        universe_info: UniverseInfo<'tcx>,
    ) -> Self {
        Self { type_checker, locations, category, universe_info }
    }
}

impl<'tcx> TypeRelatingDelegate<'tcx> for NllTypeRelatingDelegate<'_, '_, 'tcx> {
    fn span(&self) -> Span {
        self.locations.span(self.type_checker.body)
    }

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

    #[instrument(skip(self), level = "debug")]
    fn next_existential_region_var(
        &mut self,
        from_forall: bool,
        _name: Option<Symbol>,
    ) -> ty::Region<'tcx> {
        let origin = NllRegionVariableOrigin::Existential { from_forall };

        let reg_var =
            self.type_checker.infcx.next_nll_region_var(origin, || RegionCtxt::Existential(_name));

        reg_var
    }

    #[instrument(skip(self), level = "debug")]
    fn next_placeholder_region(&mut self, placeholder: ty::PlaceholderRegion) -> ty::Region<'tcx> {
        let reg = self
            .type_checker
            .borrowck_context
            .constraints
            .placeholder_region(self.type_checker.infcx, placeholder);

        let reg_info = match placeholder.name {
            ty::BoundRegionKind::BrAnon(_, Some(span)) => BoundRegionInfo::Span(span),
            ty::BoundRegionKind::BrAnon(..) => BoundRegionInfo::Name(Symbol::intern("anon")),
            ty::BoundRegionKind::BrNamed(_, name) => BoundRegionInfo::Name(name),
            ty::BoundRegionKind::BrEnv => BoundRegionInfo::Name(Symbol::intern("env")),
        };

        let reg_var =
            reg.as_var().unwrap_or_else(|| bug!("expected region {:?} to be of kind ReVar", reg));

        if cfg!(debug_assertions) && !self.type_checker.infcx.inside_canonicalization_ctxt() {
            let mut var_to_origin = self.type_checker.infcx.reg_var_to_origin.borrow_mut();
            debug!(?reg_var);
            var_to_origin.insert(reg_var, RegionCtxt::Placeholder(reg_info));
        }

        reg
    }

    #[instrument(skip(self), level = "debug")]
    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        let reg = self.type_checker.infcx.next_nll_region_var_in_universe(
            NllRegionVariableOrigin::Existential { from_forall: false },
            universe,
        );

        let reg_var =
            reg.as_var().unwrap_or_else(|| bug!("expected region {:?} to be of kind ReVar", reg));

        if cfg!(debug_assertions) && !self.type_checker.infcx.inside_canonicalization_ctxt() {
            let mut var_to_origin = self.type_checker.infcx.reg_var_to_origin.borrow_mut();
            var_to_origin.insert(reg_var, RegionCtxt::Existential(None));
        }

        reg
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
                span: self.locations.span(self.type_checker.body),
                category: self.category,
                variance_info: info,
                from_closure: false,
            },
        );
    }

    fn forbid_inference_vars() -> bool {
        true
    }

    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>) {
        self.type_checker
            .fully_perform_op(
                self.locations,
                self.category,
                InstantiateOpaqueType {
                    obligations,
                    // These fields are filled in during execution of the operation
                    base_universe: None,
                    region_constraints: None,
                },
            )
            .unwrap();
    }
}
