use rustc_middle::traits::solve::Goal;
use rustc_middle::ty::relate::combine::{super_combine_consts, super_combine_tys};
use rustc_middle::ty::relate::{
    Relate, RelateResult, TypeRelation, relate_args_invariantly, relate_args_with_variances,
};
use rustc_middle::ty::{self, DelayedSet, Ty, TyCtxt, TyVar};
use rustc_span::Span;
use tracing::{debug, instrument};

use crate::infer::BoundRegionConversionTime::HigherRankedType;
use crate::infer::relate::{PredicateEmittingRelation, StructurallyRelateAliases};
use crate::infer::{DefineOpaqueTypes, InferCtxt, SubregionOrigin, TypeTrace};
use crate::traits::{Obligation, PredicateObligations};

/// Enforce that `a` is equal to or a subtype of `b`.
pub(crate) struct TypeRelating<'infcx, 'tcx> {
    infcx: &'infcx InferCtxt<'tcx>,

    // Immutable fields
    trace: TypeTrace<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    define_opaque_types: DefineOpaqueTypes,

    // Mutable fields.
    ambient_variance: ty::Variance,
    obligations: PredicateObligations<'tcx>,
    /// The cache only tracks the `ambient_variance` as it's the
    /// only field which is mutable and which meaningfully changes
    /// the result when relating types.
    ///
    /// The cache does not track whether the state of the
    /// `InferCtxt` has been changed or whether we've added any
    /// obligations to `self.goals`. Whether a goal is added
    /// once or multiple times is not really meaningful.
    ///
    /// Changes in the inference state may delay some type inference to
    /// the next fulfillment loop. Given that this loop is already
    /// necessary, this is also not a meaningful change. Consider
    /// the following three relations:
    /// ```text
    /// Vec<?0> sub Vec<?1>
    /// ?0 eq u32
    /// Vec<?0> sub Vec<?1>
    /// ```
    /// Without a cache, the second `Vec<?0> sub Vec<?1>` would eagerly
    /// constrain `?1` to `u32`. When using the cache entry from the
    /// first time we've related these types, this only happens when
    /// later proving the `Subtype(?0, ?1)` goal from the first relation.
    cache: DelayedSet<(ty::Variance, Ty<'tcx>, Ty<'tcx>)>,
}

impl<'infcx, 'tcx> TypeRelating<'infcx, 'tcx> {
    pub(crate) fn new(
        infcx: &'infcx InferCtxt<'tcx>,
        trace: TypeTrace<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        define_opaque_types: DefineOpaqueTypes,
        ambient_variance: ty::Variance,
    ) -> TypeRelating<'infcx, 'tcx> {
        assert!(!infcx.next_trait_solver);
        TypeRelating {
            infcx,
            trace,
            param_env,
            define_opaque_types,
            ambient_variance,
            obligations: PredicateObligations::new(),
            cache: Default::default(),
        }
    }

    pub(crate) fn into_obligations(self) -> PredicateObligations<'tcx> {
        self.obligations
    }
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for TypeRelating<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn relate_item_args(
        &mut self,
        item_def_id: rustc_hir::def_id::DefId,
        a_arg: ty::GenericArgsRef<'tcx>,
        b_arg: ty::GenericArgsRef<'tcx>,
    ) -> RelateResult<'tcx, ty::GenericArgsRef<'tcx>> {
        if self.ambient_variance == ty::Invariant {
            // Avoid fetching the variance if we are in an invariant
            // context; no need, and it can induce dependency cycles
            // (e.g., #41849).
            relate_args_invariantly(self, a_arg, b_arg)
        } else {
            let tcx = self.cx();
            let opt_variances = tcx.variances_of(item_def_id);
            relate_args_with_variances(self, item_def_id, opt_variances, a_arg, b_arg, false)
        }
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        debug!(?self.ambient_variance, "new ambient variance");

        let r = if self.ambient_variance == ty::Bivariant { Ok(a) } else { self.relate(a, b) };

        self.ambient_variance = old_ambient_variance;
        r
    }

    #[instrument(skip(self), level = "trace")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        if a == b {
            return Ok(a);
        }

        let infcx = self.infcx;
        let a = infcx.shallow_resolve(a);
        let b = infcx.shallow_resolve(b);

        if self.cache.contains(&(self.ambient_variance, a, b)) {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (&ty::Infer(TyVar(a_id)), &ty::Infer(TyVar(b_id))) => {
                match self.ambient_variance {
                    ty::Covariant => {
                        // can't make progress on `A <: B` if both A and B are
                        // type variables, so record an obligation.
                        self.obligations.push(Obligation::new(
                            self.cx(),
                            self.trace.cause.clone(),
                            self.param_env,
                            ty::Binder::dummy(ty::PredicateKind::Subtype(ty::SubtypePredicate {
                                a_is_expected: true,
                                a,
                                b,
                            })),
                        ));
                    }
                    ty::Contravariant => {
                        // can't make progress on `B <: A` if both A and B are
                        // type variables, so record an obligation.
                        self.obligations.push(Obligation::new(
                            self.cx(),
                            self.trace.cause.clone(),
                            self.param_env,
                            ty::Binder::dummy(ty::PredicateKind::Subtype(ty::SubtypePredicate {
                                a_is_expected: false,
                                a: b,
                                b: a,
                            })),
                        ));
                    }
                    ty::Invariant => {
                        infcx.inner.borrow_mut().type_variables().equate(a_id, b_id);
                    }
                    ty::Bivariant => {
                        unreachable!("Expected bivariance to be handled in relate_with_variance")
                    }
                }
            }

            (&ty::Infer(TyVar(a_vid)), _) => {
                infcx.instantiate_ty_var(self, true, a_vid, self.ambient_variance, b)?;
            }
            (_, &ty::Infer(TyVar(b_vid))) => {
                infcx.instantiate_ty_var(
                    self,
                    false,
                    b_vid,
                    self.ambient_variance.xform(ty::Contravariant),
                    a,
                )?;
            }

            (
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: a_def_id, .. }),
                &ty::Alias(ty::Opaque, ty::AliasTy { def_id: b_def_id, .. }),
            ) if a_def_id == b_def_id => {
                super_combine_tys(infcx, self, a, b)?;
            }

            (&ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }), _)
            | (_, &ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }))
                if self.define_opaque_types == DefineOpaqueTypes::Yes && def_id.is_local() =>
            {
                self.register_goals(infcx.handle_opaque_type(
                    a,
                    b,
                    self.trace.cause.span,
                    self.param_env(),
                )?);
            }

            _ => {
                super_combine_tys(infcx, self, a, b)?;
            }
        }

        assert!(self.cache.insert((self.ambient_variance, a, b)));

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        let origin = SubregionOrigin::Subtype(Box::new(self.trace.clone()));

        match self.ambient_variance {
            // Subtype(&'a u8, &'b u8) => Outlives('a: 'b) => SubRegion('b, 'a)
            ty::Covariant => {
                self.infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .make_subregion(origin, b, a);
            }
            // Suptype(&'a u8, &'b u8) => Outlives('b: 'a) => SubRegion('a, 'b)
            ty::Contravariant => {
                self.infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .make_subregion(origin, a, b);
            }
            ty::Invariant => {
                self.infcx
                    .inner
                    .borrow_mut()
                    .unwrap_region_constraints()
                    .make_eqregion(origin, a, b);
            }
            ty::Bivariant => {
                unreachable!("Expected bivariance to be handled in relate_with_variance")
            }
        }

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        super_combine_consts(self.infcx, self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        if a == b {
            // Do nothing
        } else if let Some(a) = a.no_bound_vars()
            && let Some(b) = b.no_bound_vars()
        {
            self.relate(a, b)?;
        } else {
            let span = self.trace.cause.span;
            let infcx = self.infcx;

            match self.ambient_variance {
                // Checks whether `for<..> sub <: for<..> sup` holds.
                //
                // For this to hold, **all** instantiations of the super type
                // have to be a super type of **at least one** instantiation of
                // the subtype.
                //
                // This is implemented by first entering a new universe.
                // We then replace all bound variables in `sup` with placeholders,
                // and all bound variables in `sub` with inference vars.
                // We can then just relate the two resulting types as normal.
                //
                // Note: this is a subtle algorithm. For a full explanation, please see
                // the [rustc dev guide][rd]
                //
                // [rd]: https://rustc-dev-guide.rust-lang.org/borrow_check/region_inference/placeholders_and_universes.html
                ty::Covariant => {
                    infcx.enter_forall(b, |b| {
                        let a = infcx.instantiate_binder_with_fresh_vars(span, HigherRankedType, a);
                        self.relate(a, b)
                    })?;
                }
                ty::Contravariant => {
                    infcx.enter_forall(a, |a| {
                        let b = infcx.instantiate_binder_with_fresh_vars(span, HigherRankedType, b);
                        self.relate(a, b)
                    })?;
                }

                // When **equating** binders, we check that there is a 1-to-1
                // correspondence between the bound vars in both types.
                //
                // We do so by separately instantiating one of the binders with
                // placeholders and the other with inference variables and then
                // equating the instantiated types.
                //
                // We want `for<..> A == for<..> B` -- therefore we want
                // `exists<..> A == for<..> B` and `exists<..> B == for<..> A`.
                // Check if `exists<..> A == for<..> B`
                ty::Invariant => {
                    infcx.enter_forall(b, |b| {
                        let a = infcx.instantiate_binder_with_fresh_vars(span, HigherRankedType, a);
                        self.relate(a, b)
                    })?;

                    // Check if `exists<..> B == for<..> A`.
                    infcx.enter_forall(a, |a| {
                        let b = infcx.instantiate_binder_with_fresh_vars(span, HigherRankedType, b);
                        self.relate(a, b)
                    })?;
                }
                ty::Bivariant => {
                    unreachable!("Expected bivariance to be handled in relate_with_variance")
                }
            }
        }

        Ok(a)
    }
}

impl<'tcx> PredicateEmittingRelation<InferCtxt<'tcx>> for TypeRelating<'_, 'tcx> {
    fn span(&self) -> Span {
        self.trace.span()
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases {
        StructurallyRelateAliases::No
    }

    fn register_predicates(
        &mut self,
        preds: impl IntoIterator<Item: ty::Upcast<TyCtxt<'tcx>, ty::Predicate<'tcx>>>,
    ) {
        self.obligations.extend(preds.into_iter().map(|pred| {
            Obligation::new(self.infcx.tcx, self.trace.cause.clone(), self.param_env, pred)
        }))
    }

    fn register_goals(&mut self, goals: impl IntoIterator<Item = Goal<'tcx, ty::Predicate<'tcx>>>) {
        self.obligations.extend(goals.into_iter().map(|goal| {
            Obligation::new(
                self.infcx.tcx,
                self.trace.cause.clone(),
                goal.param_env,
                goal.predicate,
            )
        }))
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
                unreachable!("Expected bivariance to be handled in relate_with_variance")
            }
        })]);
    }
}
