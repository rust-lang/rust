use rustc_type_ir::error::{ExpectedFound, TypeError};
use rustc_type_ir::inherent::*;
pub use rustc_type_ir::relate::*;
use rustc_type_ir::solve::{Goal, NoSolution};
use rustc_type_ir::{self as ty, InferCtxtLike, Interner};
use tracing::{debug, instrument};

use self::combine::PredicateEmittingRelation;

pub trait RelateExt: InferCtxtLike {
    fn relate<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
    ) -> Result<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>, NoSolution>;

    fn eq_structurally_relating_aliases<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>, NoSolution>;
}

impl<Infcx: InferCtxtLike> RelateExt for Infcx {
    fn relate<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
    ) -> Result<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>, NoSolution>
    {
        let mut relate =
            SolverRelating::new(self, StructurallyRelateAliases::No, variance, param_env);
        relate.relate(lhs, rhs)?;
        Ok(relate.goals)
    }

    fn eq_structurally_relating_aliases<T: Relate<Self::Interner>>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>, NoSolution>
    {
        let mut relate =
            SolverRelating::new(self, StructurallyRelateAliases::Yes, ty::Invariant, param_env);
        relate.relate(lhs, rhs)?;
        Ok(relate.goals)
    }
}

#[allow(unused)]
/// Enforce that `a` is equal to or a subtype of `b`.
pub struct SolverRelating<'infcx, Infcx, I: Interner> {
    infcx: &'infcx Infcx,
    structurally_relate_aliases: StructurallyRelateAliases,
    ambient_variance: ty::Variance,
    param_env: I::ParamEnv,
    goals: Vec<Goal<I, I::Predicate>>,
}

impl<'infcx, Infcx, I> SolverRelating<'infcx, Infcx, I>
where
    Infcx: InferCtxtLike<Interner = I>,
    I: Interner,
{
    fn new(
        infcx: &'infcx Infcx,
        structurally_relate_aliases: StructurallyRelateAliases,
        ambient_variance: ty::Variance,
        param_env: I::ParamEnv,
    ) -> Self {
        SolverRelating {
            infcx,
            structurally_relate_aliases,
            ambient_variance,
            param_env,
            goals: vec![],
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
                Ok(a)
            }

            (ty::Infer(ty::TyVar(a_vid)), _) => {
                infcx.instantiate_ty_var_raw(
                    self,
                    self.structurally_relate_aliases,
                    true,
                    a_vid,
                    self.ambient_variance,
                    b,
                )?;
                Ok(a)
            }
            (_, ty::Infer(ty::TyVar(b_vid))) => {
                infcx.instantiate_ty_var_raw(
                    self,
                    self.structurally_relate_aliases,
                    false,
                    b_vid,
                    self.ambient_variance.xform(ty::Contravariant),
                    a,
                )?;
                Ok(a)
            }

            (ty::Error(e), _) | (_, ty::Error(e)) => {
                infcx.set_tainted_by_errors(e);
                Ok(Ty::new_error(self.cx(), e))
            }

            // Relate integral variables to other types
            (ty::Infer(ty::IntVar(a_id)), ty::Infer(ty::IntVar(b_id))) => {
                infcx.equate_int_vids_raw(a_id, b_id);
                Ok(a)
            }
            (ty::Infer(ty::IntVar(v_id)), ty::Int(v)) => {
                infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::IntType(v));
                Ok(a)
            }
            (ty::Int(v), ty::Infer(ty::IntVar(v_id))) => {
                infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::IntType(v));
                Ok(a)
            }
            (ty::Infer(ty::IntVar(v_id)), ty::Uint(v)) => {
                infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::UintType(v));
                Ok(a)
            }
            (ty::Uint(v), ty::Infer(ty::IntVar(v_id))) => {
                infcx.instantiate_int_var_raw(v_id, ty::IntVarValue::UintType(v));
                Ok(a)
            }

            // Relate floating-point variables to other types
            (ty::Infer(ty::FloatVar(a_id)), ty::Infer(ty::FloatVar(b_id))) => {
                infcx.equate_float_vids_raw(a_id, b_id);
                Ok(a)
            }
            (ty::Infer(ty::FloatVar(v_id)), ty::Float(v)) => {
                infcx.instantiate_float_var_raw(v_id, ty::FloatVarValue::Known(v));
                Ok(a)
            }
            (ty::Float(v), ty::Infer(ty::FloatVar(v_id))) => {
                infcx.instantiate_float_var_raw(v_id, ty::FloatVarValue::Known(v));
                Ok(a)
            }

            (_, ty::Alias(..)) | (ty::Alias(..), _) => match self.structurally_relate_aliases {
                StructurallyRelateAliases::Yes => structurally_relate_tys(self, a, b),
                StructurallyRelateAliases::No => {
                    self.register_alias_relate_predicate(a, b);
                    Ok(a)
                }
            },

            // All other cases of inference are errors
            (ty::Infer(_), _) | (_, ty::Infer(_)) => {
                Err(TypeError::Sorts(ExpectedFound::new(true, a, b)))
            }

            _ => structurally_relate_tys(self, a, b),
        }
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(&mut self, a: I::Region, b: I::Region) -> RelateResult<I, I::Region> {
        match self.ambient_variance {
            // Subtype(&'a u8, &'b u8) => Outlives('a: 'b) => SubRegion('b, 'a)
            ty::Covariant => {
                self.infcx.sub_regions(b, a);
            }
            // Suptype(&'a u8, &'b u8) => Outlives('b: 'a) => SubRegion('a, 'b)
            ty::Contravariant => {
                self.infcx.sub_regions(a, b);
            }
            ty::Invariant => {
                self.infcx.equate_regions(a, b);
            }
            ty::Bivariant => {
                unreachable!("Expected bivariance to be handled in relate_with_variance")
            }
        }

        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(&mut self, a: I::Const, b: I::Const) -> RelateResult<I, I::Const> {
        if a == b {
            return Ok(a);
        }

        let infcx = self.infcx;
        let a = infcx.shallow_resolve_const(a);
        let b = infcx.shallow_resolve_const(b);

        match (a.kind(), b.kind()) {
            (
                ty::ConstKind::Infer(ty::InferConst::Var(a_vid)),
                ty::ConstKind::Infer(ty::InferConst::Var(b_vid)),
            ) => {
                infcx.equate_const_vids_raw(a_vid, b_vid);
                Ok(a)
            }

            (
                ty::ConstKind::Infer(ty::InferConst::EffectVar(a_vid)),
                ty::ConstKind::Infer(ty::InferConst::EffectVar(b_vid)),
            ) => {
                infcx.equate_effect_vids_raw(a_vid, b_vid);
                Ok(a)
            }

            // All other cases of inference with other variables are errors.
            (
                ty::ConstKind::Infer(ty::InferConst::Var(_) | ty::InferConst::EffectVar(_)),
                ty::ConstKind::Infer(_),
            )
            | (
                ty::ConstKind::Infer(_),
                ty::ConstKind::Infer(ty::InferConst::Var(_) | ty::InferConst::EffectVar(_)),
            ) => {
                panic!(
                    "tried to combine ConstKind::Infer/ConstKind::Infer(InferConst::Var): {a:?} and {b:?}"
                )
            }

            (ty::ConstKind::Infer(ty::InferConst::Var(vid)), _) => {
                infcx.instantiate_const_var_raw(
                    self,
                    self.structurally_relate_aliases,
                    true,
                    vid,
                    b,
                )?;
                Ok(b)
            }

            (_, ty::ConstKind::Infer(ty::InferConst::Var(vid))) => {
                infcx.instantiate_const_var_raw(
                    self,
                    self.structurally_relate_aliases,
                    false,
                    vid,
                    a,
                )?;
                Ok(a)
            }

            (ty::ConstKind::Infer(ty::InferConst::EffectVar(vid)), _) => {
                infcx.instantiate_effect_var_raw(vid, b);
                Ok(a)
            }

            (_, ty::ConstKind::Infer(ty::InferConst::EffectVar(vid))) => {
                infcx.instantiate_effect_var_raw(vid, a);
                Ok(a)
            }

            (ty::ConstKind::Unevaluated(..), _) | (_, ty::ConstKind::Unevaluated(..)) => {
                match self.structurally_relate_aliases {
                    StructurallyRelateAliases::No => {
                        self.register_predicates([ty::PredicateKind::AliasRelate(
                            a.into(),
                            b.into(),
                            ty::AliasRelationDirection::Equate,
                        )]);

                        Ok(b)
                    }
                    StructurallyRelateAliases::Yes => structurally_relate_consts(self, a, b),
                }
            }

            _ => structurally_relate_consts(self, a, b),
        }
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
        if let Some(a_inner) = a.no_bound_vars() {
            if let Some(b_inner) = b.no_bound_vars() {
                self.relate(a_inner, b_inner)?;
                return Ok(a);
            }
        };

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
