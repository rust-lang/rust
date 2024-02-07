//! There are four type combiners: [Equate], [Sub], [Lub], and [Glb].
//! Each implements the trait [TypeRelation] and contains methods for
//! combining two instances of various things and yielding a new instance.
//! These combiner methods always yield a `Result<T>`. To relate two
//! types, you can use `infcx.at(cause, param_env)` which then allows
//! you to use the relevant methods of [At](crate::infer::at::At).
//!
//! Combiners mostly do their specific behavior and then hand off the
//! bulk of the work to [InferCtxt::super_combine_tys] and
//! [InferCtxt::super_combine_consts].
//!
//! Combining two types may have side-effects on the inference contexts
//! which can be undone by using snapshots. You probably want to use
//! either [InferCtxt::commit_if_ok] or [InferCtxt::probe].
//!
//! On success, the  LUB/GLB operations return the appropriate bound. The
//! return value of `Equate` or `Sub` shouldn't really be used.
//!
//! ## Contravariance
//!
//! We explicitly track which argument is expected using
//! [TypeRelation::a_is_expected], so when dealing with contravariance
//! this should be correctly updated.

use super::equate::Equate;
use super::generalize::{self, CombineDelegate, Generalization};
use super::glb::Glb;
use super::lub::Lub;
use super::sub::Sub;
use crate::infer::{DefineOpaqueTypes, InferCtxt, TypeTrace};
use crate::traits::{Obligation, PredicateObligations};
use rustc_middle::infer::canonical::OriginalQueryValues;
use rustc_middle::infer::unify_key::{ConstVariableValue, EffectVarValue};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::relate::{RelateResult, TypeRelation};
use rustc_middle::ty::{self, InferConst, ToPredicate, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::ty::{AliasRelationDirection, TyVar};
use rustc_middle::ty::{IntType, UintType};

#[derive(Clone)]
pub struct CombineFields<'infcx, 'tcx> {
    pub infcx: &'infcx InferCtxt<'tcx>,
    pub trace: TypeTrace<'tcx>,
    pub cause: Option<ty::relate::Cause>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub obligations: PredicateObligations<'tcx>,
    pub define_opaque_types: DefineOpaqueTypes,
}

impl<'tcx> InferCtxt<'tcx> {
    pub fn super_combine_tys<R>(
        &self,
        relation: &mut R,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        R: ObligationEmittingRelation<'tcx>,
    {
        let a_is_expected = relation.a_is_expected();
        debug_assert!(!a.has_escaping_bound_vars());
        debug_assert!(!b.has_escaping_bound_vars());

        match (a.kind(), b.kind()) {
            // Relate integral variables to other types
            (&ty::Infer(ty::IntVar(a_id)), &ty::Infer(ty::IntVar(b_id))) => {
                self.inner
                    .borrow_mut()
                    .int_unification_table()
                    .unify_var_var(a_id, b_id)
                    .map_err(|e| int_unification_error(a_is_expected, e))?;
                Ok(a)
            }
            (&ty::Infer(ty::IntVar(v_id)), &ty::Int(v)) => {
                self.unify_integral_variable(a_is_expected, v_id, IntType(v))
            }
            (&ty::Int(v), &ty::Infer(ty::IntVar(v_id))) => {
                self.unify_integral_variable(!a_is_expected, v_id, IntType(v))
            }
            (&ty::Infer(ty::IntVar(v_id)), &ty::Uint(v)) => {
                self.unify_integral_variable(a_is_expected, v_id, UintType(v))
            }
            (&ty::Uint(v), &ty::Infer(ty::IntVar(v_id))) => {
                self.unify_integral_variable(!a_is_expected, v_id, UintType(v))
            }

            // Relate floating-point variables to other types
            (&ty::Infer(ty::FloatVar(a_id)), &ty::Infer(ty::FloatVar(b_id))) => {
                self.inner
                    .borrow_mut()
                    .float_unification_table()
                    .unify_var_var(a_id, b_id)
                    .map_err(|e| float_unification_error(a_is_expected, e))?;
                Ok(a)
            }
            (&ty::Infer(ty::FloatVar(v_id)), &ty::Float(v)) => {
                self.unify_float_variable(a_is_expected, v_id, v)
            }
            (&ty::Float(v), &ty::Infer(ty::FloatVar(v_id))) => {
                self.unify_float_variable(!a_is_expected, v_id, v)
            }

            // We don't expect `TyVar` or `Fresh*` vars at this point with lazy norm.
            (ty::Alias(..), ty::Infer(ty::TyVar(_))) | (ty::Infer(ty::TyVar(_)), ty::Alias(..))
                if self.next_trait_solver() =>
            {
                bug!(
                    "We do not expect to encounter `TyVar` this late in combine \
                    -- they should have been handled earlier"
                )
            }
            (_, ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)))
            | (ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)), _)
                if self.next_trait_solver() =>
            {
                bug!("We do not expect to encounter `Fresh` variables in the new solver")
            }

            (_, ty::Alias(..)) | (ty::Alias(..), _) if self.next_trait_solver() => {
                relation.register_type_relate_obligation(a, b);
                Ok(a)
            }

            // All other cases of inference are errors
            (&ty::Infer(_), _) | (_, &ty::Infer(_)) => {
                Err(TypeError::Sorts(ty::relate::expected_found(relation, a, b)))
            }

            // During coherence, opaque types should be treated as *possibly*
            // equal to any other type (except for possibly itself). This is an
            // extremely heavy hammer, but can be relaxed in a fowards-compatible
            // way later.
            (&ty::Alias(ty::Opaque, _), _) | (_, &ty::Alias(ty::Opaque, _)) if self.intercrate => {
                relation.register_predicates([ty::Binder::dummy(ty::PredicateKind::Ambiguous)]);
                Ok(a)
            }

            _ => ty::relate::structurally_relate_tys(relation, a, b),
        }
    }

    pub fn super_combine_consts<R>(
        &self,
        relation: &mut R,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>>
    where
        R: ObligationEmittingRelation<'tcx>,
    {
        debug!("{}.consts({:?}, {:?})", relation.tag(), a, b);
        debug_assert!(!a.has_escaping_bound_vars());
        debug_assert!(!b.has_escaping_bound_vars());
        if a == b {
            return Ok(a);
        }

        let a = self.shallow_resolve(a);
        let b = self.shallow_resolve(b);

        // We should never have to relate the `ty` field on `Const` as it is checked elsewhere that consts have the
        // correct type for the generic param they are an argument for. However there have been a number of cases
        // historically where asserting that the types are equal has found bugs in the compiler so this is valuable
        // to check even if it is a bit nasty impl wise :(
        //
        // This probe is probably not strictly necessary but it seems better to be safe and not accidentally find
        // ourselves with a check to find bugs being required for code to compile because it made inference progress.
        self.probe(|_| {
            if a.ty() == b.ty() {
                return;
            }

            // We don't have access to trait solving machinery in `rustc_infer` so the logic for determining if the
            // two const param's types are able to be equal has to go through a canonical query with the actual logic
            // in `rustc_trait_selection`.
            let canonical = self.canonicalize_query(
                relation.param_env().and((a.ty(), b.ty())),
                &mut OriginalQueryValues::default(),
            );
            self.tcx.check_tys_might_be_eq(canonical).unwrap_or_else(|_| {
                // The error will only be reported later. If we emit an ErrorGuaranteed
                // here, then we will never get to the code that actually emits the error.
                self.tcx.dcx().delayed_bug(format!(
                    "cannot relate consts of different types (a={a:?}, b={b:?})",
                ));
                // We treat these constants as if they were of the same type, so that any
                // such constants being used in impls make these impls match barring other mismatches.
                // This helps with diagnostics down the road.
            });
        });

        match (a.kind(), b.kind()) {
            (
                ty::ConstKind::Infer(InferConst::Var(a_vid)),
                ty::ConstKind::Infer(InferConst::Var(b_vid)),
            ) => {
                self.inner.borrow_mut().const_unification_table().union(a_vid, b_vid);
                return Ok(a);
            }

            (
                ty::ConstKind::Infer(InferConst::EffectVar(a_vid)),
                ty::ConstKind::Infer(InferConst::EffectVar(b_vid)),
            ) => {
                self.inner.borrow_mut().effect_unification_table().union(a_vid, b_vid);
                return Ok(a);
            }

            // All other cases of inference with other variables are errors.
            (
                ty::ConstKind::Infer(InferConst::Var(_) | InferConst::EffectVar(_)),
                ty::ConstKind::Infer(_),
            )
            | (
                ty::ConstKind::Infer(_),
                ty::ConstKind::Infer(InferConst::Var(_) | InferConst::EffectVar(_)),
            ) => {
                bug!(
                    "tried to combine ConstKind::Infer/ConstKind::Infer(InferConst::Var): {a:?} and {b:?}"
                )
            }

            (ty::ConstKind::Infer(InferConst::Var(vid)), _) => {
                return self.unify_const_variable(vid, b, relation.param_env());
            }

            (_, ty::ConstKind::Infer(InferConst::Var(vid))) => {
                return self.unify_const_variable(vid, a, relation.param_env());
            }

            (ty::ConstKind::Infer(InferConst::EffectVar(vid)), _) => {
                return Ok(self.unify_effect_variable(vid, b));
            }

            (_, ty::ConstKind::Infer(InferConst::EffectVar(vid))) => {
                return Ok(self.unify_effect_variable(vid, a));
            }

            (ty::ConstKind::Unevaluated(..), _) | (_, ty::ConstKind::Unevaluated(..))
                if self.tcx.features().generic_const_exprs || self.next_trait_solver() =>
            {
                let (a, b) = if relation.a_is_expected() { (a, b) } else { (b, a) };

                relation.register_predicates([ty::Binder::dummy(if self.next_trait_solver() {
                    ty::PredicateKind::AliasRelate(
                        a.into(),
                        b.into(),
                        ty::AliasRelationDirection::Equate,
                    )
                } else {
                    ty::PredicateKind::ConstEquate(a, b)
                })]);

                return Ok(b);
            }
            _ => {}
        }

        ty::relate::structurally_relate_consts(relation, a, b)
    }

    /// Unifies the const variable `target_vid` with the given constant.
    ///
    /// This also tests if the given const `ct` contains an inference variable which was previously
    /// unioned with `target_vid`. If this is the case, inferring `target_vid` to `ct`
    /// would result in an infinite type as we continuously replace an inference variable
    /// in `ct` with `ct` itself.
    ///
    /// This is especially important as unevaluated consts use their parents generics.
    /// They therefore often contain unused args, making these errors far more likely.
    ///
    /// A good example of this is the following:
    ///
    /// ```compile_fail,E0308
    /// #![feature(generic_const_exprs)]
    ///
    /// fn bind<const N: usize>(value: [u8; N]) -> [u8; 3 + 4] {
    ///     todo!()
    /// }
    ///
    /// fn main() {
    ///     let mut arr = Default::default();
    ///     arr = bind(arr);
    /// }
    /// ```
    ///
    /// Here `3 + 4` ends up as `ConstKind::Unevaluated` which uses the generics
    /// of `fn bind` (meaning that its args contain `N`).
    ///
    /// `bind(arr)` now infers that the type of `arr` must be `[u8; N]`.
    /// The assignment `arr = bind(arr)` now tries to equate `N` with `3 + 4`.
    ///
    /// As `3 + 4` contains `N` in its args, this must not succeed.
    ///
    /// See `tests/ui/const-generics/occurs-check/` for more examples where this is relevant.
    #[instrument(level = "debug", skip(self))]
    fn unify_const_variable(
        &self,
        target_vid: ty::ConstVid,
        ct: ty::Const<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        let span = match self.inner.borrow_mut().const_unification_table().probe_value(target_vid) {
            ConstVariableValue::Known { value } => {
                bug!("instantiating a known const var: {target_vid:?} {value} {ct}")
            }
            ConstVariableValue::Unknown { origin, universe: _ } => origin.span,
        };
        // FIXME(generic_const_exprs): Occurs check failures for unevaluated
        // constants and generic expressions are not yet handled correctly.
        let Generalization { value_may_be_infer: value, needs_wf: _ } = generalize::generalize(
            self,
            &mut CombineDelegate { infcx: self, span },
            ct,
            target_vid,
            ty::Variance::Invariant,
        )?;

        self.inner
            .borrow_mut()
            .const_unification_table()
            .union_value(target_vid, ConstVariableValue::Known { value });
        Ok(value)
    }

    fn unify_integral_variable(
        &self,
        vid_is_expected: bool,
        vid: ty::IntVid,
        val: ty::IntVarValue,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        self.inner
            .borrow_mut()
            .int_unification_table()
            .unify_var_value(vid, Some(val))
            .map_err(|e| int_unification_error(vid_is_expected, e))?;
        match val {
            IntType(v) => Ok(Ty::new_int(self.tcx, v)),
            UintType(v) => Ok(Ty::new_uint(self.tcx, v)),
        }
    }

    fn unify_float_variable(
        &self,
        vid_is_expected: bool,
        vid: ty::FloatVid,
        val: ty::FloatTy,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        self.inner
            .borrow_mut()
            .float_unification_table()
            .unify_var_value(vid, Some(ty::FloatVarValue(val)))
            .map_err(|e| float_unification_error(vid_is_expected, e))?;
        Ok(Ty::new_float(self.tcx, val))
    }

    fn unify_effect_variable(&self, vid: ty::EffectVid, val: ty::Const<'tcx>) -> ty::Const<'tcx> {
        self.inner
            .borrow_mut()
            .effect_unification_table()
            .union_value(vid, EffectVarValue::Known(val));
        val
    }
}

impl<'infcx, 'tcx> CombineFields<'infcx, 'tcx> {
    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    pub fn equate<'a>(&'a mut self, a_is_expected: bool) -> Equate<'a, 'infcx, 'tcx> {
        Equate::new(self, a_is_expected)
    }

    pub fn sub<'a>(&'a mut self, a_is_expected: bool) -> Sub<'a, 'infcx, 'tcx> {
        Sub::new(self, a_is_expected)
    }

    pub fn lub<'a>(&'a mut self, a_is_expected: bool) -> Lub<'a, 'infcx, 'tcx> {
        Lub::new(self, a_is_expected)
    }

    pub fn glb<'a>(&'a mut self, a_is_expected: bool) -> Glb<'a, 'infcx, 'tcx> {
        Glb::new(self, a_is_expected)
    }

    /// Here, `dir` is either `EqTo`, `SubtypeOf`, or `SupertypeOf`.
    /// The idea is that we should ensure that the type `a_ty` is equal
    /// to, a subtype of, or a supertype of (respectively) the type
    /// to which `b_vid` is bound.
    ///
    /// Since `b_vid` has not yet been instantiated with a type, we
    /// will first instantiate `b_vid` with a *generalized* version
    /// of `a_ty`. Generalization introduces other inference
    /// variables wherever subtyping could occur.
    #[instrument(skip(self), level = "debug")]
    pub fn instantiate(
        &mut self,
        a_ty: Ty<'tcx>,
        ambient_variance: ty::Variance,
        b_vid: ty::TyVid,
        a_is_expected: bool,
    ) -> RelateResult<'tcx, ()> {
        // Get the actual variable that b_vid has been inferred to
        debug_assert!(self.infcx.inner.borrow_mut().type_variables().probe(b_vid).is_unknown());

        // Generalize type of `a_ty` appropriately depending on the
        // direction. As an example, assume:
        //
        // - `a_ty == &'x ?1`, where `'x` is some free region and `?1` is an
        //   inference variable,
        // - and `dir` == `SubtypeOf`.
        //
        // Then the generalized form `b_ty` would be `&'?2 ?3`, where
        // `'?2` and `?3` are fresh region/type inference
        // variables. (Down below, we will relate `a_ty <: b_ty`,
        // adding constraints like `'x: '?2` and `?1 <: ?3`.)
        let Generalization { value_may_be_infer: b_ty, needs_wf } = generalize::generalize(
            self.infcx,
            &mut CombineDelegate { infcx: self.infcx, span: self.trace.span() },
            a_ty,
            b_vid,
            ambient_variance,
        )?;

        // Constrain `b_vid` to the generalized type `b_ty`.
        if let &ty::Infer(TyVar(b_ty_vid)) = b_ty.kind() {
            self.infcx.inner.borrow_mut().type_variables().equate(b_vid, b_ty_vid);
        } else {
            self.infcx.inner.borrow_mut().type_variables().instantiate(b_vid, b_ty);
        }

        if needs_wf {
            self.obligations.push(Obligation::new(
                self.tcx(),
                self.trace.cause.clone(),
                self.param_env,
                ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::WellFormed(
                    b_ty.into(),
                ))),
            ));
        }

        // Finally, relate `b_ty` to `a_ty`, as described in previous comment.
        //
        // FIXME(#16847): This code is non-ideal because all these subtype
        // relations wind up attributed to the same spans. We need
        // to associate causes/spans with each of the relations in
        // the stack to get this right.
        if b_ty.is_ty_var() {
            // This happens for cases like `<?0 as Trait>::Assoc == ?0`.
            // We can't instantiate `?0` here as that would result in a
            // cyclic type. We instead delay the unification in case
            // the alias can be normalized to something which does not
            // mention `?0`.
            if self.infcx.next_trait_solver() {
                let (lhs, rhs, direction) = match ambient_variance {
                    ty::Variance::Invariant => {
                        (a_ty.into(), b_ty.into(), AliasRelationDirection::Equate)
                    }
                    ty::Variance::Covariant => {
                        (a_ty.into(), b_ty.into(), AliasRelationDirection::Subtype)
                    }
                    ty::Variance::Contravariant => {
                        (b_ty.into(), a_ty.into(), AliasRelationDirection::Subtype)
                    }
                    ty::Variance::Bivariant => unreachable!("bivariant generalization"),
                };
                self.obligations.push(Obligation::new(
                    self.tcx(),
                    self.trace.cause.clone(),
                    self.param_env,
                    ty::PredicateKind::AliasRelate(lhs, rhs, direction),
                ));
            } else {
                match a_ty.kind() {
                    &ty::Alias(ty::Projection, data) => {
                        // FIXME: This does not handle subtyping correctly, we could
                        // instead create a new inference variable for `a_ty`, emitting
                        // `Projection(a_ty, a_infer)` and `a_infer <: b_ty`.
                        self.obligations.push(Obligation::new(
                            self.tcx(),
                            self.trace.cause.clone(),
                            self.param_env,
                            ty::ProjectionPredicate { projection_ty: data, term: b_ty.into() },
                        ))
                    }
                    // The old solver only accepts projection predicates for associated types.
                    ty::Alias(ty::Inherent | ty::Weak | ty::Opaque, _) => {
                        return Err(TypeError::CyclicTy(a_ty));
                    }
                    _ => bug!("generalizated `{a_ty:?} to infer, not an alias"),
                }
            }
        } else {
            match ambient_variance {
                ty::Variance::Invariant => self.equate(a_is_expected).relate(a_ty, b_ty),
                ty::Variance::Covariant => self.sub(a_is_expected).relate(a_ty, b_ty),
                ty::Variance::Contravariant => self.sub(a_is_expected).relate_with_variance(
                    ty::Contravariant,
                    ty::VarianceDiagInfo::default(),
                    a_ty,
                    b_ty,
                ),
                ty::Variance::Bivariant => unreachable!("bivariant generalization"),
            }?;
        }

        Ok(())
    }

    pub fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>) {
        self.obligations.extend(obligations);
    }

    pub fn register_predicates(&mut self, obligations: impl IntoIterator<Item: ToPredicate<'tcx>>) {
        self.obligations.extend(obligations.into_iter().map(|to_pred| {
            Obligation::new(self.infcx.tcx, self.trace.cause.clone(), self.param_env, to_pred)
        }))
    }
}

pub trait ObligationEmittingRelation<'tcx>: TypeRelation<'tcx> {
    fn param_env(&self) -> ty::ParamEnv<'tcx>;

    /// Register obligations that must hold in order for this relation to hold
    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>);

    /// Register predicates that must hold in order for this relation to hold. Uses
    /// a default obligation cause, [`ObligationEmittingRelation::register_obligations`] should
    /// be used if control over the obligation causes is required.
    fn register_predicates(&mut self, obligations: impl IntoIterator<Item: ToPredicate<'tcx>>);

    /// Register an obligation that both types must be related to each other according to
    /// the [`ty::AliasRelationDirection`] given by [`ObligationEmittingRelation::alias_relate_direction`]
    fn register_type_relate_obligation(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) {
        self.register_predicates([ty::Binder::dummy(ty::PredicateKind::AliasRelate(
            a.into(),
            b.into(),
            self.alias_relate_direction(),
        ))]);
    }

    /// Relation direction emitted for `AliasRelate` predicates, corresponding to the direction
    /// of the relation.
    fn alias_relate_direction(&self) -> ty::AliasRelationDirection;
}

fn int_unification_error<'tcx>(
    a_is_expected: bool,
    v: (ty::IntVarValue, ty::IntVarValue),
) -> TypeError<'tcx> {
    let (a, b) = v;
    TypeError::IntMismatch(ExpectedFound::new(a_is_expected, a, b))
}

fn float_unification_error<'tcx>(
    a_is_expected: bool,
    v: (ty::FloatVarValue, ty::FloatVarValue),
) -> TypeError<'tcx> {
    let (ty::FloatVarValue(a), ty::FloatVarValue(b)) = v;
    TypeError::FloatMismatch(ExpectedFound::new(a_is_expected, a, b))
}
