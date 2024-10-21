use std::ops::Deref;

use rustc_type_ir::fold::TypeFoldable;
use rustc_type_ir::solve::{Certainty, Goal, NoSolution, SolverMode};
use rustc_type_ir::{self as ty, InferCtxtLike, Interner};

pub trait SolverDelegate: Deref<Target = <Self as SolverDelegate>::Infcx> + Sized {
    type Infcx: InferCtxtLike<Interner = <Self as SolverDelegate>::Interner>;
    type Interner: Interner;
    fn cx(&self) -> Self::Interner {
        (**self).cx()
    }

    type Span: Copy;

    fn build_with_canonical<V>(
        cx: Self::Interner,
        solver_mode: SolverMode,
        canonical: &ty::CanonicalQueryInput<Self::Interner, V>,
    ) -> (Self, V, ty::CanonicalVarValues<Self::Interner>)
    where
        V: TypeFoldable<Self::Interner>;

    fn fresh_var_for_kind_with_span(
        &self,
        arg: <Self::Interner as Interner>::GenericArg,
        span: Self::Span,
    ) -> <Self::Interner as Interner>::GenericArg;

    // FIXME: Uplift the leak check into this crate.
    fn leak_check(&self, max_input_universe: ty::UniverseIndex) -> Result<(), NoSolution>;

    fn try_const_eval_resolve(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        unevaluated: ty::UnevaluatedConst<Self::Interner>,
    ) -> Option<<Self::Interner as Interner>::Const>;

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
