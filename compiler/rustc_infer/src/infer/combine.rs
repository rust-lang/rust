//! There are four type combiners: [Equate], [Sub], [Lub], and [Glb].
//! Each implements the trait [TypeRelation] and contains methods for
//! combining two instances of various things and yielding a new instance.
//! These combiner methods always yield a `Result<T>`. To relate two
//! types, you can use `infcx.at(cause, param_env)` which then allows
//! you to use the relevant methods of [At](super::at::At).
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
use super::glb::Glb;
use super::lub::Lub;
use super::sub::Sub;
use super::type_variable::TypeVariableValue;
use super::{InferCtxt, MiscVariable, TypeTrace};
use crate::traits::{Obligation, PredicateObligations};
use rustc_data_structures::sso::SsoHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::infer::unify_key::{ConstVarValue, ConstVariableValue};
use rustc_middle::infer::unify_key::{ConstVariableOrigin, ConstVariableOriginKind};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, InferConst, ToPredicate, Ty, TyCtxt, TypeVisitable};
use rustc_middle::ty::{IntType, UintType};
use rustc_span::{Span, DUMMY_SP};

#[derive(Clone)]
pub struct CombineFields<'infcx, 'tcx> {
    pub infcx: &'infcx InferCtxt<'infcx, 'tcx>,
    pub trace: TypeTrace<'tcx>,
    pub cause: Option<ty::relate::Cause>,
    pub param_env: ty::ParamEnv<'tcx>,
    pub obligations: PredicateObligations<'tcx>,
    /// Whether we should define opaque types
    /// or just treat them opaquely.
    /// Currently only used to prevent predicate
    /// matching from matching anything against opaque
    /// types.
    pub define_opaque_types: bool,
}

#[derive(Copy, Clone, Debug)]
pub enum RelationDir {
    SubtypeOf,
    SupertypeOf,
    EqTo,
}

impl<'infcx, 'tcx> InferCtxt<'infcx, 'tcx> {
    pub fn super_combine_tys<R>(
        &self,
        relation: &mut R,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        R: TypeRelation<'tcx>,
    {
        let a_is_expected = relation.a_is_expected();

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
                    .map_err(|e| float_unification_error(relation.a_is_expected(), e))?;
                Ok(a)
            }
            (&ty::Infer(ty::FloatVar(v_id)), &ty::Float(v)) => {
                self.unify_float_variable(a_is_expected, v_id, v)
            }
            (&ty::Float(v), &ty::Infer(ty::FloatVar(v_id))) => {
                self.unify_float_variable(!a_is_expected, v_id, v)
            }

            // All other cases of inference are errors
            (&ty::Infer(_), _) | (_, &ty::Infer(_)) => {
                Err(TypeError::Sorts(ty::relate::expected_found(relation, a, b)))
            }

            _ => ty::relate::super_relate_tys(relation, a, b),
        }
    }

    pub fn super_combine_consts<R>(
        &self,
        relation: &mut R,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>>
    where
        R: ConstEquateRelation<'tcx>,
    {
        debug!("{}.consts({:?}, {:?})", relation.tag(), a, b);
        if a == b {
            return Ok(a);
        }

        let a = self.shallow_resolve(a);
        let b = self.shallow_resolve(b);

        let a_is_expected = relation.a_is_expected();

        match (a.kind(), b.kind()) {
            (
                ty::ConstKind::Infer(InferConst::Var(a_vid)),
                ty::ConstKind::Infer(InferConst::Var(b_vid)),
            ) => {
                self.inner
                    .borrow_mut()
                    .const_unification_table()
                    .unify_var_var(a_vid, b_vid)
                    .map_err(|e| const_unification_error(a_is_expected, e))?;
                return Ok(a);
            }

            // All other cases of inference with other variables are errors.
            (ty::ConstKind::Infer(InferConst::Var(_)), ty::ConstKind::Infer(_))
            | (ty::ConstKind::Infer(_), ty::ConstKind::Infer(InferConst::Var(_))) => {
                bug!("tried to combine ConstKind::Infer/ConstKind::Infer(InferConst::Var)")
            }

            (ty::ConstKind::Infer(InferConst::Var(vid)), _) => {
                return self.unify_const_variable(relation.param_env(), vid, b, a_is_expected);
            }

            (_, ty::ConstKind::Infer(InferConst::Var(vid))) => {
                return self.unify_const_variable(relation.param_env(), vid, a, !a_is_expected);
            }
            (ty::ConstKind::Unevaluated(..), _) if self.tcx.lazy_normalization() => {
                // FIXME(#59490): Need to remove the leak check to accommodate
                // escaping bound variables here.
                if !a.has_escaping_bound_vars() && !b.has_escaping_bound_vars() {
                    relation.const_equate_obligation(a, b);
                }
                return Ok(b);
            }
            (_, ty::ConstKind::Unevaluated(..)) if self.tcx.lazy_normalization() => {
                // FIXME(#59490): Need to remove the leak check to accommodate
                // escaping bound variables here.
                if !a.has_escaping_bound_vars() && !b.has_escaping_bound_vars() {
                    relation.const_equate_obligation(a, b);
                }
                return Ok(a);
            }
            _ => {}
        }

        ty::relate::super_relate_consts(relation, a, b)
    }

    /// Unifies the const variable `target_vid` with the given constant.
    ///
    /// This also tests if the given const `ct` contains an inference variable which was previously
    /// unioned with `target_vid`. If this is the case, inferring `target_vid` to `ct`
    /// would result in an infinite type as we continuously replace an inference variable
    /// in `ct` with `ct` itself.
    ///
    /// This is especially important as unevaluated consts use their parents generics.
    /// They therefore often contain unused substs, making these errors far more likely.
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
    /// of `fn bind` (meaning that its substs contain `N`).
    ///
    /// `bind(arr)` now infers that the type of `arr` must be `[u8; N]`.
    /// The assignment `arr = bind(arr)` now tries to equate `N` with `3 + 4`.
    ///
    /// As `3 + 4` contains `N` in its substs, this must not succeed.
    ///
    /// See `src/test/ui/const-generics/occurs-check/` for more examples where this is relevant.
    #[instrument(level = "debug", skip(self))]
    fn unify_const_variable(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        target_vid: ty::ConstVid<'tcx>,
        ct: ty::Const<'tcx>,
        vid_is_expected: bool,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        let (for_universe, span) = {
            let mut inner = self.inner.borrow_mut();
            let variable_table = &mut inner.const_unification_table();
            let var_value = variable_table.probe_value(target_vid);
            match var_value.val {
                ConstVariableValue::Known { value } => {
                    bug!("instantiating {:?} which has a known value {:?}", target_vid, value)
                }
                ConstVariableValue::Unknown { universe } => (universe, var_value.origin.span),
            }
        };
        let value = ConstInferUnifier { infcx: self, span, param_env, for_universe, target_vid }
            .relate(ct, ct)?;

        self.inner
            .borrow_mut()
            .const_unification_table()
            .unify_var_value(
                target_vid,
                ConstVarValue {
                    origin: ConstVariableOrigin {
                        kind: ConstVariableOriginKind::ConstInference,
                        span: DUMMY_SP,
                    },
                    val: ConstVariableValue::Known { value },
                },
            )
            .map(|()| value)
            .map_err(|e| const_unification_error(vid_is_expected, e))
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
            IntType(v) => Ok(self.tcx.mk_mach_int(v)),
            UintType(v) => Ok(self.tcx.mk_mach_uint(v)),
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
        Ok(self.tcx.mk_mach_float(val))
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
        dir: RelationDir,
        b_vid: ty::TyVid,
        a_is_expected: bool,
    ) -> RelateResult<'tcx, ()> {
        use self::RelationDir::*;

        // Get the actual variable that b_vid has been inferred to
        debug_assert!(self.infcx.inner.borrow_mut().type_variables().probe(b_vid).is_unknown());

        // Generalize type of `a_ty` appropriately depending on the
        // direction.  As an example, assume:
        //
        // - `a_ty == &'x ?1`, where `'x` is some free region and `?1` is an
        //   inference variable,
        // - and `dir` == `SubtypeOf`.
        //
        // Then the generalized form `b_ty` would be `&'?2 ?3`, where
        // `'?2` and `?3` are fresh region/type inference
        // variables. (Down below, we will relate `a_ty <: b_ty`,
        // adding constraints like `'x: '?2` and `?1 <: ?3`.)
        let Generalization { ty: b_ty, needs_wf } = self.generalize(a_ty, b_vid, dir)?;
        debug!(?b_ty);
        self.infcx.inner.borrow_mut().type_variables().instantiate(b_vid, b_ty);

        if needs_wf {
            self.obligations.push(Obligation::new(
                self.trace.cause.clone(),
                self.param_env,
                ty::Binder::dummy(ty::PredicateKind::WellFormed(b_ty.into()))
                    .to_predicate(self.infcx.tcx),
            ));
        }

        // Finally, relate `b_ty` to `a_ty`, as described in previous comment.
        //
        // FIXME(#16847): This code is non-ideal because all these subtype
        // relations wind up attributed to the same spans. We need
        // to associate causes/spans with each of the relations in
        // the stack to get this right.
        match dir {
            EqTo => self.equate(a_is_expected).relate(a_ty, b_ty),
            SubtypeOf => self.sub(a_is_expected).relate(a_ty, b_ty),
            SupertypeOf => self.sub(a_is_expected).relate_with_variance(
                ty::Contravariant,
                ty::VarianceDiagInfo::default(),
                a_ty,
                b_ty,
            ),
        }?;

        Ok(())
    }

    /// Attempts to generalize `ty` for the type variable `for_vid`.
    /// This checks for cycle -- that is, whether the type `ty`
    /// references `for_vid`. The `dir` is the "direction" for which we
    /// a performing the generalization (i.e., are we producing a type
    /// that can be used as a supertype etc).
    ///
    /// Preconditions:
    ///
    /// - `for_vid` is a "root vid"
    #[instrument(skip(self), level = "trace", ret)]
    fn generalize(
        &self,
        ty: Ty<'tcx>,
        for_vid: ty::TyVid,
        dir: RelationDir,
    ) -> RelateResult<'tcx, Generalization<'tcx>> {
        // Determine the ambient variance within which `ty` appears.
        // The surrounding equation is:
        //
        //     ty [op] ty2
        //
        // where `op` is either `==`, `<:`, or `:>`. This maps quite
        // naturally.
        let ambient_variance = match dir {
            RelationDir::EqTo => ty::Invariant,
            RelationDir::SubtypeOf => ty::Covariant,
            RelationDir::SupertypeOf => ty::Contravariant,
        };

        trace!(?ambient_variance);

        let for_universe = match self.infcx.inner.borrow_mut().type_variables().probe(for_vid) {
            v @ TypeVariableValue::Known { .. } => {
                bug!("instantiating {:?} which has a known value {:?}", for_vid, v,)
            }
            TypeVariableValue::Unknown { universe } => universe,
        };

        trace!(?for_universe);
        trace!(?self.trace);

        let mut generalize = Generalizer {
            infcx: self.infcx,
            cause: &self.trace.cause,
            for_vid_sub_root: self.infcx.inner.borrow_mut().type_variables().sub_root_var(for_vid),
            for_universe,
            ambient_variance,
            needs_wf: false,
            root_ty: ty,
            param_env: self.param_env,
            cache: SsoHashMap::new(),
        };

        let ty = generalize.relate(ty, ty)?;
        let needs_wf = generalize.needs_wf;
        Ok(Generalization { ty, needs_wf })
    }

    pub fn add_const_equate_obligation(
        &mut self,
        a_is_expected: bool,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) {
        let predicate = if a_is_expected {
            ty::PredicateKind::ConstEquate(a, b)
        } else {
            ty::PredicateKind::ConstEquate(b, a)
        };
        self.obligations.push(Obligation::new(
            self.trace.cause.clone(),
            self.param_env,
            ty::Binder::dummy(predicate).to_predicate(self.tcx()),
        ));
    }
}

struct Generalizer<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,

    /// The span, used when creating new type variables and things.
    cause: &'cx ObligationCause<'tcx>,

    /// The vid of the type variable that is in the process of being
    /// instantiated; if we find this within the type we are folding,
    /// that means we would have created a cyclic type.
    for_vid_sub_root: ty::TyVid,

    /// The universe of the type variable that is in the process of
    /// being instantiated. Any fresh variables that we create in this
    /// process should be in that same universe.
    for_universe: ty::UniverseIndex,

    /// Track the variance as we descend into the type.
    ambient_variance: ty::Variance,

    /// See the field `needs_wf` in `Generalization`.
    needs_wf: bool,

    /// The root type that we are generalizing. Used when reporting cycles.
    root_ty: Ty<'tcx>,

    param_env: ty::ParamEnv<'tcx>,

    cache: SsoHashMap<Ty<'tcx>, Ty<'tcx>>,
}

/// Result from a generalization operation. This includes
/// not only the generalized type, but also a bool flag
/// indicating whether further WF checks are needed.
#[derive(Debug)]
struct Generalization<'tcx> {
    ty: Ty<'tcx>,

    /// If true, then the generalized type may not be well-formed,
    /// even if the source type is well-formed, so we should add an
    /// additional check to enforce that it is. This arises in
    /// particular around 'bivariant' type parameters that are only
    /// constrained by a where-clause. As an example, imagine a type:
    ///
    ///     struct Foo<A, B> where A: Iterator<Item = B> {
    ///         data: A
    ///     }
    ///
    /// here, `A` will be covariant, but `B` is
    /// unconstrained. However, whatever it is, for `Foo` to be WF, it
    /// must be equal to `A::Item`. If we have an input `Foo<?A, ?B>`,
    /// then after generalization we will wind up with a type like
    /// `Foo<?C, ?D>`. When we enforce that `Foo<?A, ?B> <: Foo<?C,
    /// ?D>` (or `>:`), we will wind up with the requirement that `?A
    /// <: ?C`, but no particular relationship between `?B` and `?D`
    /// (after all, we do not know the variance of the normalized form
    /// of `A::Item` with respect to `A`). If we do nothing else, this
    /// may mean that `?D` goes unconstrained (as in #41677). So, in
    /// this scenario where we create a new type variable in a
    /// bivariant context, we set the `needs_wf` flag to true. This
    /// will force the calling code to check that `WF(Foo<?C, ?D>)`
    /// holds, which in turn implies that `?C::Item == ?D`. So once
    /// `?C` is constrained, that should suffice to restrict `?D`.
    needs_wf: bool,
}

impl<'tcx> TypeRelation<'tcx> for Generalizer<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn tag(&self) -> &'static str {
        "Generalizer"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        Ok(a.rebind(self.relate(a.skip_binder(), b.skip_binder())?))
    }

    fn relate_item_substs(
        &mut self,
        item_def_id: DefId,
        a_subst: SubstsRef<'tcx>,
        b_subst: SubstsRef<'tcx>,
    ) -> RelateResult<'tcx, SubstsRef<'tcx>> {
        if self.ambient_variance == ty::Variance::Invariant {
            // Avoid fetching the variance if we are in an invariant
            // context; no need, and it can induce dependency cycles
            // (e.g., #41849).
            relate::relate_substs(self, a_subst, b_subst)
        } else {
            let tcx = self.tcx();
            let opt_variances = tcx.variances_of(item_def_id);
            relate::relate_substs_with_variances(
                self,
                item_def_id,
                &opt_variances,
                a_subst,
                b_subst,
            )
        }
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);

        let result = self.relate(a, b);
        self.ambient_variance = old_ambient_variance;
        result
    }

    fn tys(&mut self, t: Ty<'tcx>, t2: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        assert_eq!(t, t2); // we are abusing TypeRelation here; both LHS and RHS ought to be ==

        if let Some(&result) = self.cache.get(&t) {
            return Ok(result);
        }
        debug!("generalize: t={:?}", t);

        // Check to see whether the type we are generalizing references
        // any other type variable related to `vid` via
        // subtyping. This is basically our "occurs check", preventing
        // us from creating infinitely sized types.
        let result = match *t.kind() {
            ty::Infer(ty::TyVar(vid)) => {
                let vid = self.infcx.inner.borrow_mut().type_variables().root_var(vid);
                let sub_vid = self.infcx.inner.borrow_mut().type_variables().sub_root_var(vid);
                if sub_vid == self.for_vid_sub_root {
                    // If sub-roots are equal, then `for_vid` and
                    // `vid` are related via subtyping.
                    Err(TypeError::CyclicTy(self.root_ty))
                } else {
                    let probe = self.infcx.inner.borrow_mut().type_variables().probe(vid);
                    match probe {
                        TypeVariableValue::Known { value: u } => {
                            debug!("generalize: known value {:?}", u);
                            self.relate(u, u)
                        }
                        TypeVariableValue::Unknown { universe } => {
                            match self.ambient_variance {
                                // Invariant: no need to make a fresh type variable.
                                ty::Invariant => {
                                    if self.for_universe.can_name(universe) {
                                        return Ok(t);
                                    }
                                }

                                // Bivariant: make a fresh var, but we
                                // may need a WF predicate. See
                                // comment on `needs_wf` field for
                                // more info.
                                ty::Bivariant => self.needs_wf = true,

                                // Co/contravariant: this will be
                                // sufficiently constrained later on.
                                ty::Covariant | ty::Contravariant => (),
                            }

                            let origin =
                                *self.infcx.inner.borrow_mut().type_variables().var_origin(vid);
                            let new_var_id = self
                                .infcx
                                .inner
                                .borrow_mut()
                                .type_variables()
                                .new_var(self.for_universe, origin);
                            let u = self.tcx().mk_ty_var(new_var_id);

                            // Record that we replaced `vid` with `new_var_id` as part of a generalization
                            // operation. This is needed to detect cyclic types. To see why, see the
                            // docs in the `type_variables` module.
                            self.infcx.inner.borrow_mut().type_variables().sub(vid, new_var_id);
                            debug!("generalize: replacing original vid={:?} with new={:?}", vid, u);
                            Ok(u)
                        }
                    }
                }
            }
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_)) => {
                // No matter what mode we are in,
                // integer/floating-point types must be equal to be
                // relatable.
                Ok(t)
            }
            _ => relate::super_relate_tys(self, t, t),
        }?;

        self.cache.insert(t, result);
        Ok(result)
    }

    fn regions(
        &mut self,
        r: ty::Region<'tcx>,
        r2: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        assert_eq!(r, r2); // we are abusing TypeRelation here; both LHS and RHS ought to be ==

        debug!("generalize: regions r={:?}", r);

        match *r {
            // Never make variables for regions bound within the type itself,
            // nor for erased regions.
            ty::ReLateBound(..) | ty::ReErased => {
                return Ok(r);
            }

            ty::RePlaceholder(..)
            | ty::ReVar(..)
            | ty::ReStatic
            | ty::ReEarlyBound(..)
            | ty::ReFree(..) => {
                // see common code below
            }
        }

        // If we are in an invariant context, we can re-use the region
        // as is, unless it happens to be in some universe that we
        // can't name. (In the case of a region *variable*, we could
        // use it if we promoted it into our universe, but we don't
        // bother.)
        if let ty::Invariant = self.ambient_variance {
            let r_universe = self.infcx.universe_of_region(r);
            if self.for_universe.can_name(r_universe) {
                return Ok(r);
            }
        }

        // FIXME: This is non-ideal because we don't give a
        // very descriptive origin for this region variable.
        Ok(self.infcx.next_region_var_in_universe(MiscVariable(self.cause.span), self.for_universe))
    }

    fn consts(
        &mut self,
        c: ty::Const<'tcx>,
        c2: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        assert_eq!(c, c2); // we are abusing TypeRelation here; both LHS and RHS ought to be ==

        match c.kind() {
            ty::ConstKind::Infer(InferConst::Var(vid)) => {
                let mut inner = self.infcx.inner.borrow_mut();
                let variable_table = &mut inner.const_unification_table();
                let var_value = variable_table.probe_value(vid);
                match var_value.val {
                    ConstVariableValue::Known { value: u } => {
                        drop(inner);
                        self.relate(u, u)
                    }
                    ConstVariableValue::Unknown { universe } => {
                        if self.for_universe.can_name(universe) {
                            Ok(c)
                        } else {
                            let new_var_id = variable_table.new_key(ConstVarValue {
                                origin: var_value.origin,
                                val: ConstVariableValue::Unknown { universe: self.for_universe },
                            });
                            Ok(self.tcx().mk_const_var(new_var_id, c.ty()))
                        }
                    }
                }
            }
            ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs, promoted }) => {
                assert_eq!(promoted, None);
                let substs = self.relate_with_variance(
                    ty::Variance::Invariant,
                    ty::VarianceDiagInfo::default(),
                    substs,
                    substs,
                )?;
                Ok(self.tcx().mk_const(ty::ConstS {
                    ty: c.ty(),
                    kind: ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs, promoted }),
                }))
            }
            _ => relate::super_relate_consts(self, c, c),
        }
    }
}

pub trait ConstEquateRelation<'tcx>: TypeRelation<'tcx> {
    /// Register an obligation that both constants must be equal to each other.
    ///
    /// If they aren't equal then the relation doesn't hold.
    fn const_equate_obligation(&mut self, a: ty::Const<'tcx>, b: ty::Const<'tcx>);
}

pub fn const_unification_error<'tcx>(
    a_is_expected: bool,
    (a, b): (ty::Const<'tcx>, ty::Const<'tcx>),
) -> TypeError<'tcx> {
    TypeError::ConstMismatch(ExpectedFound::new(a_is_expected, a, b))
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

struct ConstInferUnifier<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,

    span: Span,

    param_env: ty::ParamEnv<'tcx>,

    for_universe: ty::UniverseIndex,

    /// The vid of the const variable that is in the process of being
    /// instantiated; if we find this within the const we are folding,
    /// that means we would have created a cyclic const.
    target_vid: ty::ConstVid<'tcx>,
}

// We use `TypeRelation` here to propagate `RelateResult` upwards.
//
// Both inputs are expected to be the same.
impl<'tcx> TypeRelation<'tcx> for ConstInferUnifier<'_, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn tag(&self) -> &'static str {
        "ConstInferUnifier"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _variance: ty::Variance,
        _info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        // We don't care about variance here.
        self.relate(a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        Ok(a.rebind(self.relate(a.skip_binder(), b.skip_binder())?))
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn tys(&mut self, t: Ty<'tcx>, _t: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug_assert_eq!(t, _t);

        match t.kind() {
            &ty::Infer(ty::TyVar(vid)) => {
                let vid = self.infcx.inner.borrow_mut().type_variables().root_var(vid);
                let probe = self.infcx.inner.borrow_mut().type_variables().probe(vid);
                match probe {
                    TypeVariableValue::Known { value: u } => {
                        debug!("ConstOccursChecker: known value {:?}", u);
                        self.tys(u, u)
                    }
                    TypeVariableValue::Unknown { universe } => {
                        if self.for_universe.can_name(universe) {
                            return Ok(t);
                        }

                        let origin =
                            *self.infcx.inner.borrow_mut().type_variables().var_origin(vid);
                        let new_var_id = self
                            .infcx
                            .inner
                            .borrow_mut()
                            .type_variables()
                            .new_var(self.for_universe, origin);
                        Ok(self.tcx().mk_ty_var(new_var_id))
                    }
                }
            }
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_)) => Ok(t),
            _ => relate::super_relate_tys(self, t, t),
        }
    }

    fn regions(
        &mut self,
        r: ty::Region<'tcx>,
        _r: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug_assert_eq!(r, _r);
        debug!("ConstInferUnifier: r={:?}", r);

        match *r {
            // Never make variables for regions bound within the type itself,
            // nor for erased regions.
            ty::ReLateBound(..) | ty::ReErased => {
                return Ok(r);
            }

            ty::RePlaceholder(..)
            | ty::ReVar(..)
            | ty::ReStatic
            | ty::ReEarlyBound(..)
            | ty::ReFree(..) => {
                // see common code below
            }
        }

        let r_universe = self.infcx.universe_of_region(r);
        if self.for_universe.can_name(r_universe) {
            return Ok(r);
        } else {
            // FIXME: This is non-ideal because we don't give a
            // very descriptive origin for this region variable.
            Ok(self.infcx.next_region_var_in_universe(MiscVariable(self.span), self.for_universe))
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn consts(
        &mut self,
        c: ty::Const<'tcx>,
        _c: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        debug_assert_eq!(c, _c);

        match c.kind() {
            ty::ConstKind::Infer(InferConst::Var(vid)) => {
                // Check if the current unification would end up
                // unifying `target_vid` with a const which contains
                // an inference variable which is unioned with `target_vid`.
                //
                // Not doing so can easily result in stack overflows.
                if self
                    .infcx
                    .inner
                    .borrow_mut()
                    .const_unification_table()
                    .unioned(self.target_vid, vid)
                {
                    return Err(TypeError::CyclicConst(c));
                }

                let var_value =
                    self.infcx.inner.borrow_mut().const_unification_table().probe_value(vid);
                match var_value.val {
                    ConstVariableValue::Known { value: u } => self.consts(u, u),
                    ConstVariableValue::Unknown { universe } => {
                        if self.for_universe.can_name(universe) {
                            Ok(c)
                        } else {
                            let new_var_id =
                                self.infcx.inner.borrow_mut().const_unification_table().new_key(
                                    ConstVarValue {
                                        origin: var_value.origin,
                                        val: ConstVariableValue::Unknown {
                                            universe: self.for_universe,
                                        },
                                    },
                                );
                            Ok(self.tcx().mk_const_var(new_var_id, c.ty()))
                        }
                    }
                }
            }
            ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs, promoted }) => {
                assert_eq!(promoted, None);
                let substs = self.relate_with_variance(
                    ty::Variance::Invariant,
                    ty::VarianceDiagInfo::default(),
                    substs,
                    substs,
                )?;
                Ok(self.tcx().mk_const(ty::ConstS {
                    ty: c.ty(),
                    kind: ty::ConstKind::Unevaluated(ty::Unevaluated { def, substs, promoted }),
                }))
            }
            _ => relate::super_relate_consts(self, c, c),
        }
    }
}
