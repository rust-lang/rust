use std::fmt::Debug;
use std::ops::Deref;

use rustc_type_ir::solve::{
    Certainty, ComputeGoalFastPathOutcome, FetchEligibleAssocItemResponse, Goal, NoSolution,
    VisibleForLeakCheck,
};
use rustc_type_ir::{self as ty, CanonicalizerState, InferCtxtLike, Interner, TypeFoldable};

/// `SolverDelegate` is one of the two traits in the `rustc_type_ir` shared abstraction layer
/// between rustc and rust-analyzer abstracting over the [InferCtxt][inferctxt-doc], which had to be
/// split due to coherence reasons:
/// - `SolverDelegate` contains the parts depending on trait-solving logic, to provide functionality
///   in `rustc_trait_selection`, and is implemented by a [simple wrapper over
///   `InferCtxt`][inferctxt-wrapper-doc] there,
/// - [InferCtxtLike] contains the other parts, and is implemented [directly on
///   `InferCtxt`][inferctxtlike-impl-doc].
///
/// More information can also be found in the dedicated chapter in the dev-guide, in [this
/// section][dev-guide].
///
/// [inferctxt-doc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html
/// [inferctxt-wrapper-doc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/solve/delegate/struct.SolverDelegate.html
/// [inferctxtlike-impl-doc]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_infer/infer/struct.InferCtxt.html#impl-InferCtxtLike-for-InferCtxt%3C'tcx%3E
/// [dev-guide]: https://rustc-dev-guide.rust-lang.org/solve/sharing-crates-with-rust-analyzer.html#trait-inferctxtlike-and-trait-solverdelegate
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
    ) -> ComputeGoalFastPathOutcome<Self::Interner>;

    fn fresh_var_for_kind(
        &self,
        arg: <Self::Interner as Interner>::GenericArg,
        span: <Self::Interner as Interner>::Span,
        universe: ty::UniverseIndex,
    ) -> <Self::Interner as Interner>::GenericArg;

    // FIXME: Uplift the leak check into this crate.
    fn leak_check(&self, max_input_universe: ty::UniverseIndex) -> Result<(), NoSolution>;

    /// Evaluate a const, normalizing the type of the resulting value with `normalize_ty`.
    /// Returns `Ok(None)` if the const is too generic, and `Err(_)` only if `normalize_ty`
    /// failed.
    fn evaluate_const<E: Debug>(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        alias_const: ty::AliasConst<Self::Interner>,
        normalize_ty: impl FnOnce(
            ty::Unnormalized<Self::Interner, <Self::Interner as Interner>::Ty>,
        ) -> Result<<Self::Interner as Interner>::Ty, E>,
    ) -> Result<Option<<Self::Interner as Interner>::Const>, E>;

    // FIXME: This only is here because `wf::obligations` is in `rustc_trait_selection`!
    fn well_formed_goals(
        &self,
        param_env: <Self::Interner as Interner>::ParamEnv,
        term: <Self::Interner as Interner>::Term,
    ) -> Option<Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>>;

    fn make_deduplicated_region_constraints(
        &self,
    ) -> Vec<(ty::RegionConstraint<Self::Interner>, VisibleForLeakCheck)>;

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
        def_id: <Self::Interner as Interner>::OpaqueTyId,
        args: <Self::Interner as Interner>::GenericArgs,
        param_env: <Self::Interner as Interner>::ParamEnv,
        hidden_ty: <Self::Interner as Interner>::Ty,
        goals: &mut Vec<Goal<Self::Interner, <Self::Interner as Interner>::Predicate>>,
    );

    fn fetch_eligible_assoc_item(
        &self,
        goal_trait_ref: ty::TraitRef<Self::Interner>,
        trait_assoc_def_id: <Self::Interner as Interner>::TraitAssocTermId,
        impl_def_id: <Self::Interner as Interner>::ImplId,
    ) -> FetchEligibleAssocItemResponse<Self::Interner>;

    fn is_transmutable(
        &self,
        src: <Self::Interner as Interner>::Ty,
        dst: <Self::Interner as Interner>::Ty,
        assume: <Self::Interner as Interner>::Const,
    ) -> Result<Certainty, NoSolution>;

    /// Obtain canonicalizer state, either by allocating it afresh (the default) or by reusing
    /// previously allocated state.
    fn obtain_canonicalizer_state(&self) -> CanonicalizerState<Self::Interner> {
        Default::default()
    }

    /// Release canonicalizer state, either by deallocating it (the default) or by clearing it and
    /// stashing it for later reuse.
    fn release_canonicalizer_state(&self, _: CanonicalizerState<Self::Interner>) {}

    fn emit_next_solver_overflow_fcw(
        &self,
        predicate: <Self::Interner as Interner>::Predicate,
        span: <Self::Interner as Interner>::Span,
    );
}
