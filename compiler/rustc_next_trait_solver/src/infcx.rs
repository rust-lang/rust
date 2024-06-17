use std::fmt::Debug;

use rustc_type_ir::fold::TypeFoldable;
use rustc_type_ir::relate::Relate;
use rustc_type_ir::solve::{Certainty, Goal, NoSolution, SolverMode};
use rustc_type_ir::{self as ty, Interner};

pub trait SolverDelegate: Sized {
    type Interner: Interner;
    fn interner(&self) -> Self::Interner;

    type Span: Copy;

    fn solver_mode(&self) -> SolverMode;

    fn build_with_canonical<V>(
        interner: Self::Interner,
        solver_mode: SolverMode,
        canonical: &ty::Canonical<Self::Interner, V>,
    ) -> (Self, V, ty::CanonicalVarValues<Self::Interner>)
    where
        V: TypeFoldable<Self::Interner>;

    fn universe(&self) -> ty::UniverseIndex;
    fn create_next_universe(&self) -> ty::UniverseIndex;

    fn universe_of_ty(&self, ty: ty::TyVid) -> Option<ty::UniverseIndex>;
    fn universe_of_lt(&self, lt: ty::RegionVid) -> Option<ty::UniverseIndex>;
    fn universe_of_ct(&self, ct: ty::ConstVid) -> Option<ty::UniverseIndex>;

    fn root_ty_var(&self, var: ty::TyVid) -> ty::TyVid;
    fn root_const_var(&self, var: ty::ConstVid) -> ty::ConstVid;

    fn opportunistic_resolve_ty_var(&self, vid: ty::TyVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_int_var(&self, vid: ty::IntVid) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_float_var(
        &self,
        vid: ty::FloatVid,
    ) -> <Self::Interner as Interner>::Ty;
    fn opportunistic_resolve_ct_var(
        &self,
        vid: ty::ConstVid,
    ) -> <Self::Interner as Interner>::Const;
    fn opportunistic_resolve_effect_var(
        &self,
        vid: ty::EffectVid,
    ) -> <Self::Interner as Interner>::Const;
    fn opportunistic_resolve_lt_var(
        &self,
        vid: ty::RegionVid,
    ) -> <Self::Interner as Interner>::Region;

    fn defining_opaque_types(&self) -> <Self::Interner as Interner>::DefiningOpaqueTypes;

    fn next_ty_infer(&self) -> <Self::Interner as Interner>::Ty;
    fn next_const_infer(&self) -> <Self::Interner as Interner>::Const;
    fn fresh_args_for_item(
        &self,
        def_id: <Self::Interner as Interner>::DefId,
    ) -> <Self::Interner as Interner>::GenericArgs;

    fn fresh_var_for_kind_with_span(
        &self,
        arg: <Self::Interner as Interner>::GenericArg,
        span: Self::Span,
    ) -> <Self::Interner as Interner>::GenericArg;

    fn instantiate_binder_with_infer<T: TypeFoldable<Self::Interner> + Copy>(
        &self,
        value: ty::Binder<Self::Interner, T>,
    ) -> T;

    fn enter_forall<T: TypeFoldable<Self::Interner> + Copy, U>(
        &self,
        value: ty::Binder<Self::Interner, T>,
        f: impl FnOnce(T) -> U,
    ) -> U;

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

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<Self::Interner>;

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T;

    // FIXME: Uplift the leak check into this crate.
    fn leak_check(&self, max_input_universe: ty::UniverseIndex) -> Result<(), NoSolution>;

    // FIXME: This is only here because elaboration lives in `rustc_infer`!
    fn elaborate_supertraits(
        interner: Self::Interner,
        trait_ref: ty::Binder<Self::Interner, ty::TraitRef<Self::Interner>>,
    ) -> impl Iterator<Item = ty::Binder<Self::Interner, ty::TraitRef<Self::Interner>>>;

    fn try_const_eval_resolve(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        unevaluated: ty::UnevaluatedConst<Self::Interner>,
    ) -> Option<<Self::Interner as Interner>::Const>;

    fn sub_regions(
        &self,
        sub: <Self::Interner as Interner>::Region,
        sup: <Self::Interner as Interner>::Region,
    );

    fn register_ty_outlives(
        &self,
        ty: <Self::Interner as Interner>::Ty,
        r: <Self::Interner as Interner>::Region,
    );

    // FIXME: This only is here because `wf::obligations` is in `rustc_trait_selection`!
    fn well_formed_goals(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        arg: <Self::Interner as Interner>::GenericArg,
    ) -> Option<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>>;

    fn clone_opaque_types_for_query_response(
        &self,
    ) -> Vec<(ty::OpaqueTypeKey<Self::Interner>, <Self::Interner as Interner>::Ty)>;

    fn make_deduplicated_outlives_constraints(
        &self,
    ) -> Vec<ty::OutlivesPredicate<Self::Interner, <Self::Interner as Interner>::GenericArg>>;

    fn instantiate_canonical<V>(
        &self,
        canonical: ty::Canonical<Self::Interner, V>,
        values: ty::CanonicalVarValues<Self::Interner>,
    ) -> V
    where
        V: TypeFoldable<Self::Interner>;

    fn instantiate_canonical_var_with_infer(
        &self,
        cv_info: ty::CanonicalVarInfo<Self::Interner>,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> <Self::Interner as Interner>::GenericArg;

    // FIXME: Can we implement this in terms of `add` and `inject`?
    fn insert_hidden_type(
        &self,
        opaque_type_key: ty::OpaqueTypeKey<Self::Interner>,
        param_env: <Self::Interner as Interner>::ParamEnv,
        hidden_ty: <Self::Interner as Interner>::Ty,
        goals: &mut Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
    ) -> Result<(), NoSolution>;

    fn add_item_bounds_for_hidden_type(
        &self,
        def_id: <Self::Interner as Interner>::DefId,
        args: <Self::Interner as Interner>::GenericArgs,
        param_env: <Self::Interner as Interner>::ParamEnv,
        hidden_ty: <Self::Interner as Interner>::Ty,
        goals: &mut Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
    );

    fn inject_new_hidden_type_unchecked(
        &self,
        key: ty::OpaqueTypeKey<Self::Interner>,
        hidden_ty: <Self::Interner as Interner>::Ty,
    );

    fn reset_opaque_types(&self);

    fn trait_ref_is_knowable<E: Debug>(
        &self,
        trait_ref: ty::TraitRef<Self::Interner>,
        lazily_normalize_ty: impl FnMut(
            <Self::Interner as Interner>::Ty,
        ) -> Result<<Self::Interner as Interner>::Ty, E>,
    ) -> Result<bool, E>;

    fn fetch_eligible_assoc_item(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        goal_trait_ref: ty::TraitRef<Self::Interner>,
        trait_assoc_def_id: <Self::Interner as Interner>::DefId,
        impl_def_id: <Self::Interner as Interner>::DefId,
    ) -> Result<Option<<Self::Interner as Interner>::DefId>, NoSolution>;

    fn is_transmutable(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        dst: <Self::Interner as Interner>::Ty,
        src: <Self::Interner as Interner>::Ty,
        assume: <Self::Interner as Interner>::Const,
    ) -> Result<Certainty, NoSolution>;
}
