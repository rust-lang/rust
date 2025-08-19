//! Trait Resolution. See the [rustc-dev-guide] for more information on how this works.
//!
//! [rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

mod engine;
mod project;
mod structural_impls;
pub mod util;

use std::cmp;
use std::hash::{Hash, Hasher};

use hir::def_id::LocalDefId;
use rustc_hir as hir;
use rustc_macros::{TypeFoldable, TypeVisitable};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::Certainty;
pub use rustc_middle::traits::*;
use rustc_middle::ty::{self, Ty, TyCtxt, Upcast};
use rustc_span::Span;
use thin_vec::ThinVec;

pub use self::engine::{FromSolverError, ScrubbedTraitError, TraitEngine};
pub(crate) use self::project::UndoLog;
pub use self::project::{
    MismatchedProjectionTypes, Normalized, NormalizedTerm, ProjectionCache, ProjectionCacheEntry,
    ProjectionCacheKey, ProjectionCacheStorage,
};
use crate::infer::InferCtxt;

/// An `Obligation` represents some trait reference (e.g., `i32: Eq`) for
/// which the "impl_source" must be found. The process of finding an "impl_source" is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for i32`) that
/// satisfies the obligation, or else finding a bound that is in
/// scope. The eventual result is usually a `Selection` (defined below).
#[derive(Clone, TypeFoldable, TypeVisitable)]
pub struct Obligation<'tcx, T> {
    /// The reason we have to prove this thing.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub cause: ObligationCause<'tcx>,

    /// The environment in which we should prove this thing.
    pub param_env: ty::ParamEnv<'tcx>,

    /// The thing we are trying to prove.
    pub predicate: T,

    /// If we started proving this as a result of trying to prove
    /// something else, track the total depth to ensure termination.
    /// If this goes over a certain threshold, we abort compilation --
    /// in such cases, we can not say whether or not the predicate
    /// holds for certain. Stupid halting problem; such a drag.
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    pub recursion_depth: usize,
}

impl<'tcx, T: Copy> Obligation<'tcx, T> {
    pub fn as_goal(&self) -> solve::Goal<'tcx, T> {
        solve::Goal { param_env: self.param_env, predicate: self.predicate }
    }
}

impl<'tcx, T: PartialEq> PartialEq<Obligation<'tcx, T>> for Obligation<'tcx, T> {
    #[inline]
    fn eq(&self, other: &Obligation<'tcx, T>) -> bool {
        // Ignore `cause` and `recursion_depth`. This is a small performance
        // win for a few crates, and a huge performance win for the crate in
        // https://github.com/rust-lang/rustc-perf/pull/1680, which greatly
        // stresses the trait system.
        self.param_env == other.param_env && self.predicate == other.predicate
    }
}

impl<T: Eq> Eq for Obligation<'_, T> {}

impl<T: Hash> Hash for Obligation<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) -> () {
        // See the comment on `Obligation::eq`.
        self.param_env.hash(state);
        self.predicate.hash(state);
    }
}

pub type PredicateObligation<'tcx> = Obligation<'tcx, ty::Predicate<'tcx>>;
pub type TraitObligation<'tcx> = Obligation<'tcx, ty::TraitPredicate<'tcx>>;
pub type PolyTraitObligation<'tcx> = Obligation<'tcx, ty::PolyTraitPredicate<'tcx>>;

pub type PredicateObligations<'tcx> = ThinVec<PredicateObligation<'tcx>>;

impl<'tcx> PredicateObligation<'tcx> {
    /// Flips the polarity of the inner predicate.
    ///
    /// Given `T: Trait` predicate it returns `T: !Trait` and given `T: !Trait` returns `T: Trait`.
    pub fn flip_polarity(&self, tcx: TyCtxt<'tcx>) -> Option<PredicateObligation<'tcx>> {
        Some(PredicateObligation {
            cause: self.cause.clone(),
            param_env: self.param_env,
            predicate: self.predicate.flip_polarity(tcx)?,
            recursion_depth: self.recursion_depth,
        })
    }
}

impl<'tcx> PolyTraitObligation<'tcx> {
    pub fn derived_cause(
        &self,
        variant: impl FnOnce(DerivedCause<'tcx>) -> ObligationCauseCode<'tcx>,
    ) -> ObligationCause<'tcx> {
        self.cause.clone().derived_cause(self.predicate, variant)
    }
}

// `PredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(PredicateObligation<'_>, 48);

pub type Selection<'tcx> = ImplSource<'tcx, PredicateObligation<'tcx>>;

/// A callback that can be provided to `inspect_typeck`. Invoked on evaluation
/// of root obligations.
pub type ObligationInspector<'tcx> =
    fn(&InferCtxt<'tcx>, &PredicateObligation<'tcx>, Result<Certainty, NoSolution>);

impl<'tcx, O> Obligation<'tcx, O> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: impl Upcast<TyCtxt<'tcx>, O>,
    ) -> Obligation<'tcx, O> {
        Self::with_depth(tcx, cause, 0, param_env, predicate)
    }

    /// We often create nested obligations without setting the correct depth.
    ///
    /// To deal with this evaluate and fulfill explicitly update the depth
    /// of nested obligations using this function.
    pub fn set_depth_from_parent(&mut self, parent_depth: usize) {
        self.recursion_depth = cmp::max(parent_depth + 1, self.recursion_depth);
    }

    pub fn with_depth(
        tcx: TyCtxt<'tcx>,
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        predicate: impl Upcast<TyCtxt<'tcx>, O>,
    ) -> Obligation<'tcx, O> {
        let predicate = predicate.upcast(tcx);
        Obligation { cause, param_env, recursion_depth, predicate }
    }

    pub fn misc(
        tcx: TyCtxt<'tcx>,
        span: Span,
        body_id: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: impl Upcast<TyCtxt<'tcx>, O>,
    ) -> Obligation<'tcx, O> {
        Obligation::new(tcx, ObligationCause::misc(span, body_id), param_env, trait_ref)
    }

    pub fn with<P>(
        &self,
        tcx: TyCtxt<'tcx>,
        value: impl Upcast<TyCtxt<'tcx>, P>,
    ) -> Obligation<'tcx, P> {
        Obligation::with_depth(tcx, self.cause.clone(), self.recursion_depth, self.param_env, value)
    }
}

impl<'tcx> PolyTraitObligation<'tcx> {
    pub fn polarity(&self) -> ty::PredicatePolarity {
        self.predicate.skip_binder().polarity
    }

    pub fn self_ty(&self) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.predicate.map_bound(|p| p.self_ty())
    }
}
