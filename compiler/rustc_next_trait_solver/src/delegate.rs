use std::ops::Deref;

use rustc_type_ir::solve::{Certainty, Goal, NoSolution};
use rustc_type_ir::{self as ty, InferCtxtLike, Interner, TypeFoldable};

pub trait SolverDelegate: Deref<Target = Self::Infcx> + Sized {
    type Infcx: InferCtxtLike<Interner = Self::Interner>;
    type Interner: Interner;
    fn cx(&self) -> Self::Interner {
        (**self).cx()
    }

    fn build_with_canonical<V>(
        cx: Self::Interner,
        canonical: &ty::CanonicalQueryInput<Self::Interner, V>,
    ) -> (Self, V, ty::CanonicalVarValues<Self::Interner>)
    where
        V: TypeFoldable<Self::Interner>;

    fn compute_goal_fast_path(
        &self,
        goal: Goal<Self::Interner, <Self::Interner as Interner>::Predicate>,
        span: <Self::Interner as Interner>::Span,
    ) -> Option<Certainty>;

    fn fresh_var_for_kind_with_span(
        &self,
        arg: <Self::Interner as Interner>::GenericArg,
        span: <Self::Interner as Interner>::Span,
    ) -> <Self::Interner as Interner>::GenericArg;

    // FIXME: Uplift the leak check into this crate.
    fn leak_check(&self, max_input_universe: ty::UniverseIndex) -> Result<(), NoSolution>;

    fn evaluate_const(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        uv: ty::UnevaluatedConst<Self::Interner>,
    ) -> Option<<Self::Interner as Interner>::Const>;

    // FIXME: This only is here because `wf::obligations` is in `rustc_trait_selection`!
    fn well_formed_goals(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        term: <Self::Interner as Interner>::Term,
    ) -> Option<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>>;

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

    fn instantiate_canonical_var(
        &self,
        kind: ty::CanonicalVarKind<Self::Interner>,
        span: <Self::Interner as Interner>::Span,
        var_values: &[<Self::Interner as Interner>::GenericArg],
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> <Self::Interner as Interner>::GenericArg;

    fn add_item_bounds_for_hidden_type(
        &self,
        def_id: <Self::Interner as Interner>::DefId,
        args: <Self::Interner as Interner>::GenericArgs,
        param_env: <Self::Interner as Interner>::ParamEnv,
        hidden_ty: <Self::Interner as Interner>::Ty,
        goals: &mut Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
    );

    fn fetch_eligible_assoc_item(
        &self,
        goal_trait_ref: ty::TraitRef<Self::Interner>,
        trait_assoc_def_id: <Self::Interner as Interner>::DefId,
        impl_def_id: <Self::Interner as Interner>::ImplId,
    ) -> Result<
        Option<<Self::Interner as Interner>::DefId>,
        <Self::Interner as Interner>::ErrorGuaranteed,
    >;

    fn is_transmutable(
        &self,
        dst: <Self::Interner as Interner>::Ty,
        src: <Self::Interner as Interner>::Ty,
        assume: <Self::Interner as Interner>::Const,
    ) -> Result<Certainty, NoSolution>;
}
