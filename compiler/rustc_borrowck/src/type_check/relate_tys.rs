use rustc_data_structures::fx::FxHashMap;
use rustc_errors::ErrorGuaranteed;
use rustc_infer::infer::relate::{
    PredicateEmittingRelation, Relate, RelateResult, StructurallyRelateAliases, TypeRelation,
};
use rustc_infer::infer::{InferCtxt, NllRegionVariableOrigin};
use rustc_infer::traits::Obligation;
use rustc_infer::traits::solve::Goal;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::relate::combine::{super_combine_consts, super_combine_tys};
use rustc_middle::ty::{self, FnMutDelegate, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_span::{Span, Symbol, sym};
use tracing::{debug, instrument};

use crate::constraints::OutlivesConstraint;
use crate::diagnostics::UniverseInfo;
use crate::renumber::RegionCtxt;
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
    ) -> Result<(), NoSolution> {
        NllTypeRelating::new(self, locations, category, UniverseInfo::relate(a, b), v)
            .relate(a, b)?;
        Ok(())
    }

    /// Add sufficient constraints to ensure `a == b`. See also [Self::relate_types].
    pub(super) fn eq_args(
        &mut self,
        a: ty::GenericArgsRef<'tcx>,
        b: ty::GenericArgsRef<'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
    ) -> Result<(), NoSolution> {
        NllTypeRelating::new(self, locations, category, UniverseInfo::other(), ty::Invariant)
            .relate(a, b)?;
        Ok(())
    }
}

struct NllTypeRelating<'a, 'b, 'tcx> {
    type_checker: &'a mut TypeChecker<'b, 'tcx>,

    /// Where (and why) is this relation taking place?
    locations: Locations,

    /// What category do we assign the resulting `'a: 'b` relationships?
    category: ConstraintCategory<'tcx>,

    /// Information so that error reporting knows what types we are relating
    /// when reporting a bound region error.
    universe_info: UniverseInfo<'tcx>,

    /// How are we relating `a` and `b`?
    ///
    /// - Covariant means `a <: b`.
    /// - Contravariant means `b <: a`.
    /// - Invariant means `a == b`.
    /// - Bivariant means that it doesn't matter.
    ambient_variance: ty::Variance,

    ambient_variance_info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
}

impl<'a, 'b, 'tcx> NllTypeRelating<'a, 'b, 'tcx> {
    fn new(
        type_checker: &'a mut TypeChecker<'b, 'tcx>,
        locations: Locations,
        category: ConstraintCategory<'tcx>,
        universe_info: UniverseInfo<'tcx>,
        ambient_variance: ty::Variance,
    ) -> Self {
        Self {
            type_checker,
            locations,
            category,
            universe_info,
            ambient_variance,
            ambient_variance_info: ty::VarianceDiagInfo::default(),
        }
    }

    fn ambient_covariance(&self) -> bool {
        match self.ambient_variance {
            ty::Covariant | ty::Invariant => true,
            ty::Contravariant | ty::Bivariant => false,
        }
    }

    fn ambient_contravariance(&self) -> bool {
        match self.ambient_variance {
            ty::Contravariant | ty::Invariant => true,
            ty::Covariant | ty::Bivariant => false,
        }
    }

    fn relate_opaques(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, ()> {
        let infcx = self.type_checker.infcx;
        debug_assert!(!infcx.next_trait_solver());
        // `handle_opaque_type` cannot handle subtyping, so to support subtyping
        // we instead eagerly generalize here. This is a bit of a mess but will go
        // away once we're using the new solver.
        //
        // Given `opaque rel B`, we create a new infer var `ty_vid` constrain it
        // by using `ty_vid rel B` and then finally and end by equating `ty_vid` to
        // the opaque.
        let mut enable_subtyping = |ty, opaque_is_expected| {
            // We create the fresh inference variable in the highest universe.
            // In theory we could limit it to the highest universe in the args of
            // the opaque but that isn't really worth the effort.
            //
            // We'll make sure that the opaque type can actually name everything
            // in its hidden type later on.
            let ty_vid = infcx.next_ty_vid(self.span());
            let variance = if opaque_is_expected {
                self.ambient_variance
            } else {
                self.ambient_variance.xform(ty::Contravariant)
            };

            self.type_checker.infcx.instantiate_ty_var(
                self,
                opaque_is_expected,
                ty_vid,
                variance,
                ty,
            )?;
            Ok(infcx.resolve_vars_if_possible(Ty::new_infer(infcx.tcx, ty::TyVar(ty_vid))))
        };

        let (a, b) = match (a.kind(), b.kind()) {
            (&ty::Alias(ty::Opaque, ..), _) => (a, enable_subtyping(b, true)?),
            (_, &ty::Alias(ty::Opaque, ..)) => (enable_subtyping(a, false)?, b),
            _ => unreachable!(
                "expected at least one opaque type in `relate_opaques`, got {a} and {b}."
            ),
        };
        self.register_goals(infcx.handle_opaque_type(a, b, self.span(), self.param_env())?);
        Ok(())
    }

    fn enter_forall<T, U>(
        &mut self,
        binder: ty::Binder<'tcx, T>,
        f: impl FnOnce(&mut Self, T) -> U,
    ) -> U
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        let value = if let Some(inner) = binder.no_bound_vars() {
            inner
        } else {
            let infcx = self.type_checker.infcx;
            let mut lazy_universe = None;
            let delegate = FnMutDelegate {
                regions: &mut |br: ty::BoundRegion| {
                    // The first time this closure is called, create a
                    // new universe for the placeholders we will make
                    // from here out.
                    let universe = lazy_universe.unwrap_or_else(|| {
                        let universe = self.create_next_universe();
                        lazy_universe = Some(universe);
                        universe
                    });

                    let placeholder = ty::PlaceholderRegion { universe, bound: br };
                    debug!(?placeholder);
                    let placeholder_reg = self.next_placeholder_region(placeholder);
                    debug!(?placeholder_reg);

                    placeholder_reg
                },
                types: &mut |_bound_ty: ty::BoundTy| {
                    unreachable!("we only replace regions in nll_relate, not types")
                },
                consts: &mut |_bound_const: ty::BoundConst| {
                    unreachable!("we only replace regions in nll_relate, not consts")
                },
            };

            infcx.tcx.replace_bound_vars_uncached(binder, delegate)
        };

        debug!(?value);
        f(self, value)
    }

    #[instrument(skip(self), level = "debug")]
    fn instantiate_binder_with_existentials<T>(&mut self, binder: ty::Binder<'tcx, T>) -> T
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        if let Some(inner) = binder.no_bound_vars() {
            return inner;
        }

        let infcx = self.type_checker.infcx;
        let mut reg_map = FxHashMap::default();
        let delegate = FnMutDelegate {
            regions: &mut |br: ty::BoundRegion| {
                if let Some(ex_reg_var) = reg_map.get(&br) {
                    *ex_reg_var
                } else {
                    let ex_reg_var =
                        self.next_existential_region_var(br.kind.get_name(infcx.infcx.tcx));
                    debug!(?ex_reg_var);
                    reg_map.insert(br, ex_reg_var);

                    ex_reg_var
                }
            },
            types: &mut |_bound_ty: ty::BoundTy| {
                unreachable!("we only replace regions in nll_relate, not types")
            },
            consts: &mut |_bound_const: ty::BoundConst| {
                unreachable!("we only replace regions in nll_relate, not consts")
            },
        };

        let replaced = infcx.tcx.replace_bound_vars_uncached(binder, delegate);
        debug!(?replaced);

        replaced
    }

    fn create_next_universe(&mut self) -> ty::UniverseIndex {
        let universe = self.type_checker.infcx.create_next_universe();
        self.type_checker.constraints.universe_causes.insert(universe, self.universe_info.clone());
        universe
    }

    #[instrument(skip(self), level = "debug")]
    fn next_existential_region_var(&mut self, name: Option<Symbol>) -> ty::Region<'tcx> {
        let origin = NllRegionVariableOrigin::Existential { name };
        self.type_checker.infcx.next_nll_region_var(origin, || RegionCtxt::Existential(name))
    }

    #[instrument(skip(self), level = "debug")]
    fn next_placeholder_region(&mut self, placeholder: ty::PlaceholderRegion) -> ty::Region<'tcx> {
        let reg =
            self.type_checker.constraints.placeholder_region(self.type_checker.infcx, placeholder);

        let reg_info = match placeholder.bound.kind {
            ty::BoundRegionKind::Anon => sym::anon,
            ty::BoundRegionKind::Named(def_id) => self.type_checker.tcx().item_name(def_id),
            ty::BoundRegionKind::ClosureEnv => sym::env,
            ty::BoundRegionKind::NamedAnon(_) => bug!("only used for pretty printing"),
        };

        if cfg!(debug_assertions) {
            let mut var_to_origin = self.type_checker.infcx.reg_var_to_origin.borrow_mut();
            let new = RegionCtxt::Placeholder(reg_info);
            let prev = var_to_origin.insert(reg.as_var(), new);
            if let Some(prev) = prev {
                assert_eq!(new, prev);
            }
        }

        reg
    }

    fn push_outlives(
        &mut self,
        sup: ty::Region<'tcx>,
        sub: ty::Region<'tcx>,
        info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
    ) {
        let sub = self.type_checker.universal_regions.to_region_vid(sub);
        let sup = self.type_checker.universal_regions.to_region_vid(sup);
        self.type_checker.constraints.outlives_constraints.push(OutlivesConstraint {
            sup,
            sub,
            locations: self.locations,
            span: self.locations.span(self.type_checker.body),
            category: self.category,
            variance_info: info,
            from_closure: false,
        });
    }
}

impl<'b, 'tcx> TypeRelation<TyCtxt<'tcx>> for NllTypeRelating<'_, 'b, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.type_checker.infcx.tcx
    }

    #[instrument(skip(self, info), level = "trace", ret)]
    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        self.ambient_variance_info = self.ambient_variance_info.xform(info);

        debug!(?self.ambient_variance);
        // In a bivariant context this always succeeds.
        let r = if self.ambient_variance == ty::Bivariant { Ok(a) } else { self.relate(a, b) };

        self.ambient_variance = old_ambient_variance;

        r
    }

    #[instrument(skip(self), level = "debug")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        let infcx = self.type_checker.infcx;

        let a = self.type_checker.infcx.shallow_resolve(a);
        assert!(!b.has_non_region_infer(), "unexpected inference var {:?}", b);

        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (_, &ty::Infer(ty::TyVar(_))) => {
                span_bug!(
                    self.span(),
                    "should not be relating type variables on the right in MIR typeck"
                );
            }

            (&ty::Infer(ty::TyVar(a_vid)), _) => {
                infcx.instantiate_ty_var(self, true, a_vid, self.ambient_variance, b)?
            }

            (
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id || infcx.next_trait_solver() => {
                super_combine_tys(&infcx.infcx, self, a, b).map(|_| ()).or_else(|err| {
                    // This behavior is only there for the old solver, the new solver
                    // shouldn't ever fail. Instead, it unconditionally emits an
                    // alias-relate goal.
                    assert!(!self.type_checker.infcx.next_trait_solver());
                    self.cx().dcx().span_delayed_bug(
                        self.span(),
                        "failure to relate an opaque to itself should result in an error later on",
                    );
                    if a_def_id.is_local() { self.relate_opaques(a, b) } else { Err(err) }
                })?;
            }
            (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
            | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
                if def_id.is_local() && !self.type_checker.infcx.next_trait_solver() =>
            {
                self.relate_opaques(a, b)?;
            }

            _ => {
                debug!(?a, ?b, ?self.ambient_variance);

                // Will also handle unification of `IntVar` and `FloatVar`.
                super_combine_tys(&self.type_checker.infcx.infcx, self, a, b)?;
            }
        }

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!(?self.ambient_variance);

        if self.ambient_covariance() {
            // Covariant: &'a u8 <: &'b u8. Hence, `'a: 'b`.
            self.push_outlives(a, b, self.ambient_variance_info);
        }

        if self.ambient_contravariance() {
            // Contravariant: &'b u8 <: &'a u8. Hence, `'b: 'a`.
            self.push_outlives(b, a, self.ambient_variance_info);
        }

        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        let a = self.type_checker.infcx.shallow_resolve_const(a);
        assert!(!a.has_non_region_infer(), "unexpected inference var {:?}", a);
        assert!(!b.has_non_region_infer(), "unexpected inference var {:?}", b);

        super_combine_consts(&self.type_checker.infcx.infcx, self, a, b)
    }

    #[instrument(skip(self), level = "trace")]
    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        // We want that
        //
        // ```
        // for<'a> fn(&'a u32) -> &'a u32 <:
        //   fn(&'b u32) -> &'b u32
        // ```
        //
        // but not
        //
        // ```
        // fn(&'a u32) -> &'a u32 <:
        //   for<'b> fn(&'b u32) -> &'b u32
        // ```
        //
        // We therefore proceed as follows:
        //
        // - Instantiate binders on `b` universally, yielding a universe U1.
        // - Instantiate binders on `a` existentially in U1.

        debug!(?self.ambient_variance);

        if let (Some(a), Some(b)) = (a.no_bound_vars(), b.no_bound_vars()) {
            // Fast path for the common case.
            self.relate(a, b)?;
            return Ok(ty::Binder::dummy(a));
        }

        match self.ambient_variance {
            ty::Covariant => {
                // Covariance, so we want `for<..> A <: for<..> B` --
                // therefore we compare any instantiation of A (i.e., A
                // instantiated with existentials) against every
                // instantiation of B (i.e., B instantiated with
                // universals).

                // Note: the order here is important. Create the placeholders first, otherwise
                // we assign the wrong universe to the existential!
                self.enter_forall(b, |this, b| {
                    let a = this.instantiate_binder_with_existentials(a);
                    this.relate(a, b)
                })?;
            }

            ty::Contravariant => {
                // Contravariance, so we want `for<..> A :> for<..> B` --
                // therefore we compare every instantiation of A (i.e., A
                // instantiated with universals) against any
                // instantiation of B (i.e., B instantiated with
                // existentials). Opposite of above.

                // Note: the order here is important. Create the placeholders first, otherwise
                // we assign the wrong universe to the existential!
                self.enter_forall(a, |this, a| {
                    let b = this.instantiate_binder_with_existentials(b);
                    this.relate(a, b)
                })?;
            }

            ty::Invariant => {
                // Invariant, so we want `for<..> A == for<..> B` --
                // therefore we want `exists<..> A == for<..> B` and
                // `exists<..> B == for<..> A`.
                //
                // See the comment in `fn Equate::binders` for more details.

                // Note: the order here is important. Create the placeholders first, otherwise
                // we assign the wrong universe to the existential!
                self.enter_forall(b, |this, b| {
                    let a = this.instantiate_binder_with_existentials(a);
                    this.relate(a, b)
                })?;
                // Note: the order here is important. Create the placeholders first, otherwise
                // we assign the wrong universe to the existential!
                self.enter_forall(a, |this, a| {
                    let b = this.instantiate_binder_with_existentials(b);
                    this.relate(a, b)
                })?;
            }

            ty::Bivariant => {}
        }

        Ok(a)
    }
}

impl<'b, 'tcx> PredicateEmittingRelation<InferCtxt<'tcx>> for NllTypeRelating<'_, 'b, 'tcx> {
    fn span(&self) -> Span {
        self.locations.span(self.type_checker.body)
    }

    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases {
        StructurallyRelateAliases::No
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.type_checker.infcx.param_env
    }

    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item: ty::Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>>,
    ) {
        let tcx = self.cx();
        let param_env = self.param_env();
        self.register_goals(
            obligations.into_iter().map(|to_pred| Goal::new(tcx, param_env, to_pred)),
        );
    }

    fn register_goals(
        &mut self,
        obligations: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>,
    ) {
        let _: Result<_, ErrorGuaranteed> = self.type_checker.fully_perform_op(
            self.locations,
            self.category,
            InstantiateOpaqueType {
                obligations: obligations
                    .into_iter()
                    .map(|goal| {
                        Obligation::new(
                            self.cx(),
                            ObligationCause::dummy_with_span(self.span()),
                            goal.param_env,
                            goal.predicate,
                        )
                    })
                    .collect(),
                // These fields are filled in during execution of the operation
                base_universe: None,
                region_constraints: None,
            },
        );
    }

    fn register_alias_relate_predicate(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) {
        self.register_predicates([ty::Binder::dummy(match self.ambient_variance {
            ty::Covariant => ty::PredicateKind::AliasRelate(
                a.into(),
                b.into(),
                ty::AliasRelationDirection::Subtype,
            ),
            // a :> b is b <: a
            ty::Contravariant => ty::PredicateKind::AliasRelate(
                b.into(),
                a.into(),
                ty::AliasRelationDirection::Subtype,
            ),
            ty::Invariant => ty::PredicateKind::AliasRelate(
                a.into(),
                b.into(),
                ty::AliasRelationDirection::Equate,
            ),
            ty::Bivariant => {
                unreachable!("cannot defer an alias-relate goal with Bivariant variance (yet?)")
            }
        })]);
    }
}
