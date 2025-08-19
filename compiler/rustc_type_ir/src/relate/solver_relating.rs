use tracing::{debug, instrument};

use self::combine::{PredicateEmittingRelation, super_combine_consts, super_combine_tys};
use crate::data_structures::DelayedSet;
pub use crate::relate::*;
use crate::solve::Goal;
use crate::{self as ty, InferCtxtLike, Interner};

pub trait RelateExt: InferCtxtLike {
    fn relate<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
        span: <Self::Interner as Interner>::Span,
    ) -> Result<
        Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
        TypeError<Self::Interner>,
    >;

    fn eq_structurally_relating_aliases<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        rhs: T,
        span: <Self::Interner as Interner>::Span,
    ) -> Result<
        Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
        TypeError<Self::Interner>,
    >;
}

impl<Infcx: InferCtxtLike> RelateExt for Infcx {
    fn relate<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
        span: <Self::Interner as Interner>::Span,
    ) -> Result<
        Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
        TypeError<Self::Interner>,
    > {
        let mut relate =
            SolverRelating::new(self, StructurallyRelateAliases::No, variance, param_env, span);
        relate.relate(lhs, rhs)?;
        Ok(relate.goals)
    }

    fn eq_structurally_relating_aliases<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        rhs: T,
        span: <Self::Interner as Interner>::Span,
    ) -> Result<
        Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
        TypeError<Self::Interner>,
    > {
        let mut relate = SolverRelating::new(
            self,
            StructurallyRelateAliases::Yes,
            ty::Invariant,
            param_env,
            span,
        );
        relate.relate(lhs, rhs)?;
        Ok(relate.goals)
    }
}

/// Enforce that `a` is equal to or a subtype of `b`.
pub struct SolverRelating<'infcx, Infcx, I: Interner> {
    infcx: &'infcx Infcx,
    // Immutable fields.
    structurally_relate_aliases: StructurallyRelateAliases,
    param_env: I::ParamEnv,
    span: I::Span,
    // Mutable fields.
    ambient_variance: ty::Variance,
    goals: Vec<Goal<I, I::Predicate>>,
    /// The cache only tracks the `ambient_variance` as it's the
    /// only field which is mutable and which meaningfully changes
    /// the result when relating types.
    ///
    /// The cache does not track whether the state of the
    /// `Infcx` has been changed or whether we've added any
    /// goals to `self.goals`. Whether a goal is added once or multiple
    /// times is not really meaningful.
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
    cache: DelayedSet<(ty::Variance, I::Ty, I::Ty)>,
}

impl<'infcx, Infcx, I> SolverRelating<'infcx, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    pub fn new(
        infcx: &'infcx Infcx,
        structurally_relate_aliases: StructurallyRelateAliases,
        ambient_variance: ty::Variance,
        param_env: I::ParamEnv,
        span: I::Span,
    ) -> Self {
        SolverRelating {
            infcx,
            structurally_relate_aliases,
            span,
            ambient_variance,
            param_env,
            goals: vec![],
            cache: Default::default(),
        }
    }
}

impl<Infcx, I> TypeRelation<I> for SolverRelating<'_, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn cx(&self) -> I {
        self.infcx.cx()
    }

    fn relate_item_args(
        &mut self,
        item_def_id: I::DefId,
        a_arg: I::GenericArgs,
        b_arg: I::GenericArgs,
    ) -> RelateResult<I, I::GenericArgs> {
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

    fn relate_with_variance<T: Relate<I>>(
        &mut self,
        variance: ty::Variance,
        _info: VarianceDiagInfo<I>,
        a: T,
        b: T,
    ) -> RelateResult<I, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        debug!(?self.ambient_variance, "new ambient variance");

        let r = if self.ambient_variance == ty::Bivariant { Ok(a) } else { self.relate(a, b) };

        self.ambient_variance = old_ambient_variance;
        r
    }

    #[instrument(skip(self), level = "trace")]
    fn tys(&mut self, a: I::Ty, b: I::Ty) -> RelateResult<I, I::Ty> {
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
            (ty::Infer(ty::TyVar(a_id)), ty::Infer(ty::TyVar(b_id))) => {
                match self.ambient_variance {
                    ty::Covariant => {
                        // can't make progress on `A <: B` if both A and B are
                        // type variables, so record an obligation.
                        self.goals.push(Goal::new(
                            self.cx(),
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
                        self.goals.push(Goal::new(
                            self.cx(),
                            self.param_env,
                            ty::Binder::dummy(ty::PredicateKind::Subtype(ty::SubtypePredicate {
                                a_is_expected: false,
                                a: b,
                                b: a,
                            })),
                        ));
                    }
                    ty::Invariant => {
                        infcx.equate_ty_vids_raw(a_id, b_id);
                    }
                    ty::Bivariant => {
                        unreachable!("Expected bivariance to be handled in relate_with_variance")
                    }
                }
            }

            (ty::Infer(ty::TyVar(a_vid)), _) => {
                infcx.instantiate_ty_var_raw(self, true, a_vid, self.ambient_variance, b)?;
            }
            (_, ty::Infer(ty::TyVar(b_vid))) => {
                infcx.instantiate_ty_var_raw(
                    self,
                    false,
                    b_vid,
                    self.ambient_variance.xform(ty::Contravariant),
                    a,
                )?;
            }

            _ => {
                super_combine_tys(self.infcx, self, a, b)?;
            }
        }

        assert!(self.cache.insert((self.ambient_variance, a, b)));

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(&mut self, a: I::Region, b: I::Region) -> RelateResult<I, I::Region> {
        match self.ambient_variance {
            // Subtype(&'a u8, &'b u8) => Outlives('a: 'b) => SubRegion('b, 'a)
            ty::Covariant => self.infcx.sub_regions(b, a, self.span),
            // Suptype(&'a u8, &'b u8) => Outlives('b: 'a) => SubRegion('a, 'b)
            ty::Contravariant => self.infcx.sub_regions(a, b, self.span),
            ty::Invariant => self.infcx.equate_regions(a, b, self.span),
            ty::Bivariant => {
                unreachable!("Expected bivariance to be handled in relate_with_variance")
            }
        }

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(&mut self, a: I::Const, b: I::Const) -> RelateResult<I, I::Const> {
        super_combine_consts(self.infcx, self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<I, T>,
        b: ty::Binder<I, T>,
    ) -> RelateResult<I, ty::Binder<I, T>>
    where
        T: Relate<I>,
    {
        // If they're equal, then short-circuit.
        if a == b {
            return Ok(a);
        }

        // If they have no bound vars, relate normally.
        if let Some(a_inner) = a.no_bound_vars()
            && let Some(b_inner) = b.no_bound_vars()
        {
            self.relate(a_inner, b_inner)?;
            return Ok(a);
        }

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
                self.infcx.enter_forall(b, |b| {
                    let a = self.infcx.instantiate_binder_with_infer(a);
                    self.relate(a, b)
                })?;
            }
            ty::Contravariant => {
                self.infcx.enter_forall(a, |a| {
                    let b = self.infcx.instantiate_binder_with_infer(b);
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
                self.infcx.enter_forall(b, |b| {
                    let a = self.infcx.instantiate_binder_with_infer(a);
                    self.relate(a, b)
                })?;

                // Check if `exists<..> B == for<..> A`.
                self.infcx.enter_forall(a, |a| {
                    let b = self.infcx.instantiate_binder_with_infer(b);
                    self.relate(a, b)
                })?;
            }
            ty::Bivariant => {
                unreachable!("Expected bivariance to be handled in relate_with_variance")
            }
        }
        Ok(a)
    }
}

impl<Infcx, I> PredicateEmittingRelation<Infcx> for SolverRelating<'_, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn span(&self) -> I::Span {
        Span::dummy()
    }

    fn param_env(&self) -> I::ParamEnv {
        self.param_env
    }

    fn structurally_relate_aliases(&self) -> StructurallyRelateAliases {
        self.structurally_relate_aliases
    }

    fn register_predicates(
        &mut self,
        obligations: impl IntoIterator<Item: ty::Upcast<I, I::Predicate>>,
    ) {
        self.goals.extend(
            obligations.into_iter().map(|pred| Goal::new(self.infcx.cx(), self.param_env, pred)),
        );
    }

    fn register_goals(&mut self, obligations: impl IntoIterator<Item = Goal<I, I::Predicate>>) {
        self.goals.extend(obligations);
    }

    fn register_alias_relate_predicate(&mut self, a: I::Ty, b: I::Ty) {
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
