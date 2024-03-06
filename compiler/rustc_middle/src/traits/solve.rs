use rustc_ast_ir::try_visit;
use rustc_data_structures::intern::Interned;
use rustc_span::def_id::DefId;

use crate::infer::canonical::{CanonicalVarValues, QueryRegionConstraints};
use crate::traits::query::NoSolution;
use crate::traits::{Canonical, DefiningAnchor};
use crate::ty::{
    self, FallibleTypeFolder, ToPredicate, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeVisitable,
    TypeVisitor,
};

use super::BuiltinImplSource;

mod cache;
pub mod inspect;

pub use cache::{CacheData, EvaluationCache};

/// A goal is a statement, i.e. `predicate`, we want to prove
/// given some assumptions, i.e. `param_env`.
///
/// Most of the time the `param_env` contains the `where`-bounds of the function
/// we're currently typechecking while the `predicate` is some trait bound.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct Goal<'tcx, P> {
    pub predicate: P,
    pub param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx, P> Goal<'tcx, P> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: impl ToPredicate<'tcx, P>,
    ) -> Goal<'tcx, P> {
        Goal { param_env, predicate: predicate.to_predicate(tcx) }
    }

    /// Updates the goal to one with a different `predicate` but the same `param_env`.
    pub fn with<Q>(self, tcx: TyCtxt<'tcx>, predicate: impl ToPredicate<'tcx, Q>) -> Goal<'tcx, Q> {
        Goal { param_env: self.param_env, predicate: predicate.to_predicate(tcx) }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct Response<'tcx> {
    pub certainty: Certainty,
    pub var_values: CanonicalVarValues<'tcx>,
    /// Additional constraints returned by this query.
    pub external_constraints: ExternalConstraints<'tcx>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum Certainty {
    Yes,
    Maybe(MaybeCause),
}

impl Certainty {
    pub const AMBIGUOUS: Certainty = Certainty::Maybe(MaybeCause::Ambiguity);

    /// Use this function to merge the certainty of multiple nested subgoals.
    ///
    /// Given an impl like `impl<T: Foo + Bar> Baz for T {}`, we have 2 nested
    /// subgoals whenever we use the impl as a candidate: `T: Foo` and `T: Bar`.
    /// If evaluating `T: Foo` results in ambiguity and `T: Bar` results in
    /// success, we merge these two responses. This results in ambiguity.
    ///
    /// If we unify ambiguity with overflow, we return overflow. This doesn't matter
    /// inside of the solver as we do not distinguish ambiguity from overflow. It does
    /// however matter for diagnostics. If `T: Foo` resulted in overflow and `T: Bar`
    /// in ambiguity without changing the inference state, we still want to tell the
    /// user that `T: Baz` results in overflow.
    pub fn unify_with(self, other: Certainty) -> Certainty {
        match (self, other) {
            (Certainty::Yes, Certainty::Yes) => Certainty::Yes,
            (Certainty::Yes, Certainty::Maybe(_)) => other,
            (Certainty::Maybe(_), Certainty::Yes) => self,
            (Certainty::Maybe(a), Certainty::Maybe(b)) => Certainty::Maybe(a.unify_with(b)),
        }
    }

    pub const fn overflow(suggest_increasing_limit: bool) -> Certainty {
        Certainty::Maybe(MaybeCause::Overflow { suggest_increasing_limit })
    }
}

/// Why we failed to evaluate a goal.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub enum MaybeCause {
    /// We failed due to ambiguity. This ambiguity can either
    /// be a true ambiguity, i.e. there are multiple different answers,
    /// or we hit a case where we just don't bother, e.g. `?x: Trait` goals.
    Ambiguity,
    /// We gave up due to an overflow, most often by hitting the recursion limit.
    Overflow { suggest_increasing_limit: bool },
}

impl MaybeCause {
    fn unify_with(self, other: MaybeCause) -> MaybeCause {
        match (self, other) {
            (MaybeCause::Ambiguity, MaybeCause::Ambiguity) => MaybeCause::Ambiguity,
            (MaybeCause::Ambiguity, MaybeCause::Overflow { .. }) => other,
            (MaybeCause::Overflow { .. }, MaybeCause::Ambiguity) => self,
            (
                MaybeCause::Overflow { suggest_increasing_limit: a },
                MaybeCause::Overflow { suggest_increasing_limit: b },
            ) => MaybeCause::Overflow { suggest_increasing_limit: a || b },
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct QueryInput<'tcx, T> {
    pub goal: Goal<'tcx, T>,
    pub anchor: DefiningAnchor,
    pub predefined_opaques_in_body: PredefinedOpaques<'tcx>,
}

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, HashStable, Default)]
pub struct PredefinedOpaquesData<'tcx> {
    pub opaque_types: Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, HashStable)]
pub struct PredefinedOpaques<'tcx>(pub(crate) Interned<'tcx, PredefinedOpaquesData<'tcx>>);

impl<'tcx> std::ops::Deref for PredefinedOpaques<'tcx> {
    type Target = PredefinedOpaquesData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub type CanonicalInput<'tcx, T = ty::Predicate<'tcx>> = Canonical<'tcx, QueryInput<'tcx, T>>;

pub type CanonicalResponse<'tcx> = Canonical<'tcx, Response<'tcx>>;

/// The result of evaluating a canonical query.
///
/// FIXME: We use a different type than the existing canonical queries. This is because
/// we need to add a `Certainty` for `overflow` and may want to restructure this code without
/// having to worry about changes to currently used code. Once we've made progress on this
/// solver, merge the two responses again.
pub type QueryResult<'tcx> = Result<CanonicalResponse<'tcx>, NoSolution>;

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash, HashStable)]
pub struct ExternalConstraints<'tcx>(pub(crate) Interned<'tcx, ExternalConstraintsData<'tcx>>);

impl<'tcx> std::ops::Deref for ExternalConstraints<'tcx> {
    type Target = ExternalConstraintsData<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Additional constraints returned on success.
#[derive(Debug, PartialEq, Eq, Clone, Hash, HashStable, Default, TypeVisitable, TypeFoldable)]
pub struct ExternalConstraintsData<'tcx> {
    // FIXME: implement this.
    pub region_constraints: QueryRegionConstraints<'tcx>,
    pub opaque_types: Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)>,
}

// FIXME: Having to clone `region_constraints` for folding feels bad and
// probably isn't great wrt performance.
//
// Not sure how to fix this, maybe we should also intern `opaque_types` and
// `region_constraints` here or something.
impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().try_fold_with(folder)?,
            opaque_types: self
                .opaque_types
                .iter()
                .map(|opaque| opaque.try_fold_with(folder))
                .collect::<Result<_, F::Error>>()?,
        }))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        TypeFolder::interner(folder).mk_external_constraints(ExternalConstraintsData {
            region_constraints: self.region_constraints.clone().fold_with(folder),
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for ExternalConstraints<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        try_visit!(self.region_constraints.visit_with(visitor));
        self.opaque_types.visit_with(visitor)
    }
}

// FIXME: Having to clone `region_constraints` for folding feels bad and
// probably isn't great wrt performance.
//
// Not sure how to fix this, maybe we should also intern `opaque_types` and
// `region_constraints` here or something.
impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for PredefinedOpaques<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(FallibleTypeFolder::interner(folder).mk_predefined_opaques_in_body(
            PredefinedOpaquesData {
                opaque_types: self
                    .opaque_types
                    .iter()
                    .map(|opaque| opaque.try_fold_with(folder))
                    .collect::<Result<_, F::Error>>()?,
            },
        ))
    }

    fn fold_with<F: TypeFolder<TyCtxt<'tcx>>>(self, folder: &mut F) -> Self {
        TypeFolder::interner(folder).mk_predefined_opaques_in_body(PredefinedOpaquesData {
            opaque_types: self.opaque_types.iter().map(|opaque| opaque.fold_with(folder)).collect(),
        })
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for PredefinedOpaques<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> V::Result {
        self.opaque_types.visit_with(visitor)
    }
}

/// Why a specific goal has to be proven.
///
/// This is necessary as we treat nested goals different depending on
/// their source.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GoalSource {
    Misc,
    /// We're proving a where-bound of an impl.
    ///
    /// FIXME(-Znext-solver=coinductive): Explain how and why this
    /// changes whether cycles are coinductive.
    ///
    /// This also impacts whether we erase constraints on overflow.
    /// Erasing constraints is generally very useful for perf and also
    /// results in better error messages by avoiding spurious errors.
    /// We do not erase overflow constraints in `normalizes-to` goals unless
    /// they are from an impl where-clause. This is necessary due to
    /// backwards compatability, cc trait-system-refactor-initiatitive#70.
    ImplWhereBound,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, HashStable)]
pub enum IsNormalizesToHack {
    Yes,
    No,
}

/// Possible ways the given goal can be proven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSource {
    /// A user written impl.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// fn main() {
    ///     let x: Vec<u32> = Vec::new();
    ///     // This uses the impl from the standard library to prove `Vec<T>: Clone`.
    ///     let y = x.clone();
    /// }
    /// ```
    Impl(DefId),
    /// A builtin impl generated by the compiler. When adding a new special
    /// trait, try to use actual impls whenever possible. Builtin impls should
    /// only be used in cases where the impl cannot be manually be written.
    ///
    /// Notable examples are auto traits, `Sized`, and `DiscriminantKind`.
    /// For a list of all traits with builtin impls, check out the
    /// `EvalCtxt::assemble_builtin_impl_candidates` method.
    BuiltinImpl(BuiltinImplSource),
    /// An assumption from the environment.
    ///
    /// More precisely we've used the `n-th` assumption in the `param_env`.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// fn is_clone<T: Clone>(x: T) -> (T, T) {
    ///     // This uses the assumption `T: Clone` from the `where`-bounds
    ///     // to prove `T: Clone`.
    ///     (x.clone(), x)
    /// }
    /// ```
    ParamEnv(usize),
    /// If the self type is an alias type, e.g. an opaque type or a projection,
    /// we know the bounds on that alias to hold even without knowing its concrete
    /// underlying type.
    ///
    /// More precisely this candidate is using the `n-th` bound in the `item_bounds` of
    /// the self type.
    ///
    /// ## Examples
    ///
    /// ```rust
    /// trait Trait {
    ///     type Assoc: Clone;
    /// }
    ///
    /// fn foo<T: Trait>(x: <T as Trait>::Assoc) {
    ///     // We prove `<T as Trait>::Assoc` by looking at the bounds on `Assoc` in
    ///     // in the trait definition.
    ///     let _y = x.clone();
    /// }
    /// ```
    AliasBound,
}
