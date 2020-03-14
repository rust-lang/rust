//! Trait Resolution. See the [rustc guide] for more information on how this works.
//!
//! [rustc guide]: https://rust-lang.github.io/rustc-guide/traits/resolution.html

mod engine;
pub mod error_reporting;
mod project;
mod structural_impls;
mod util;

use rustc::ty::error::{ExpectedFound, TypeError};
use rustc::ty::{self, Ty};
use rustc_hir as hir;
use rustc_span::Span;

pub use self::FulfillmentErrorCode::*;
pub use self::ObligationCauseCode::*;
pub use self::SelectionError::*;
pub use self::Vtable::*;

pub use self::engine::{TraitEngine, TraitEngineExt};
pub use self::project::MismatchedProjectionTypes;
pub use self::project::{
    Normalized, NormalizedTy, ProjectionCache, ProjectionCacheEntry, ProjectionCacheKey,
    ProjectionCacheSnapshot, Reveal,
};
crate use self::util::elaborate_predicates;

pub use rustc::traits::*;

/// An `Obligation` represents some trait reference (e.g., `int: Eq`) for
/// which the vtable must be found. The process of finding a vtable is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for int`) that
/// provides the required vtable, or else finding a bound that is in
/// scope. The eventual result is usually a `Selection` (defined below).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Obligation<'tcx, T> {
    /// The reason we have to prove this thing.
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
    pub recursion_depth: usize,
}

pub type PredicateObligation<'tcx> = Obligation<'tcx, ty::Predicate<'tcx>>;
pub type TraitObligation<'tcx> = Obligation<'tcx, ty::PolyTraitPredicate<'tcx>>;

// `PredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(target_arch = "x86_64")]
static_assert_size!(PredicateObligation<'_>, 112);

pub type Obligations<'tcx, O> = Vec<Obligation<'tcx, O>>;
pub type PredicateObligations<'tcx> = Vec<PredicateObligation<'tcx>>;
pub type TraitObligations<'tcx> = Vec<TraitObligation<'tcx>>;

pub type Selection<'tcx> = Vtable<'tcx, PredicateObligation<'tcx>>;

pub struct FulfillmentError<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub code: FulfillmentErrorCode<'tcx>,
    /// Diagnostics only: we opportunistically change the `code.span` when we encounter an
    /// obligation error caused by a call argument. When this is the case, we also signal that in
    /// this field to ensure accuracy of suggestions.
    pub points_at_arg_span: bool,
}

#[derive(Clone)]
pub enum FulfillmentErrorCode<'tcx> {
    CodeSelectionError(SelectionError<'tcx>),
    CodeProjectionError(MismatchedProjectionTypes<'tcx>),
    CodeSubtypeError(ExpectedFound<Ty<'tcx>>, TypeError<'tcx>), // always comes from a SubtypePredicate
    CodeAmbiguity,
}

impl<'tcx, O> Obligation<'tcx, O> {
    pub fn new(
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        predicate: O,
    ) -> Obligation<'tcx, O> {
        Obligation { cause, param_env, recursion_depth: 0, predicate }
    }

    pub fn with_depth(
        cause: ObligationCause<'tcx>,
        recursion_depth: usize,
        param_env: ty::ParamEnv<'tcx>,
        predicate: O,
    ) -> Obligation<'tcx, O> {
        Obligation { cause, param_env, recursion_depth, predicate }
    }

    pub fn misc(
        span: Span,
        body_id: hir::HirId,
        param_env: ty::ParamEnv<'tcx>,
        trait_ref: O,
    ) -> Obligation<'tcx, O> {
        Obligation::new(ObligationCause::misc(span, body_id), param_env, trait_ref)
    }

    pub fn with<P>(&self, value: P) -> Obligation<'tcx, P> {
        Obligation {
            cause: self.cause.clone(),
            param_env: self.param_env,
            recursion_depth: self.recursion_depth,
            predicate: value,
        }
    }
}

impl<'tcx> FulfillmentError<'tcx> {
    pub fn new(
        obligation: PredicateObligation<'tcx>,
        code: FulfillmentErrorCode<'tcx>,
    ) -> FulfillmentError<'tcx> {
        FulfillmentError { obligation, code, points_at_arg_span: false }
    }
}

impl<'tcx> TraitObligation<'tcx> {
    pub fn self_ty(&self) -> ty::Binder<Ty<'tcx>> {
        self.predicate.map_bound(|p| p.self_ty())
    }
}
