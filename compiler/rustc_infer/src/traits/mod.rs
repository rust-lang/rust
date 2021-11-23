//! Trait Resolution. See the [rustc-dev-guide] for more information on how this works.
//!
//! [rustc-dev-guide]: https://rustc-dev-guide.rust-lang.org/traits/resolution.html

mod engine;
pub mod error_reporting;
mod project;
mod structural_impls;
pub mod util;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, Const, Ty, TyCtxt};
use rustc_span::Span;
use std::borrow::{Borrow, Cow};
use std::collections::hash_map::RawEntryMut;
use std::hash::Hash;

pub use self::FulfillmentErrorCode::*;
pub use self::ImplSource::*;
pub use self::ObligationCauseCode::*;
pub use self::SelectionError::*;

pub use self::engine::{TraitEngine, TraitEngineExt};
pub use self::project::MismatchedProjectionTypes;
pub(crate) use self::project::UndoLog;
pub use self::project::{
    Normalized, NormalizedTy, ProjectionCache, ProjectionCacheEntry, ProjectionCacheKey,
    ProjectionCacheStorage, Reveal,
};
pub use rustc_middle::traits::*;

/// An `Obligation` represents some trait reference (e.g., `i32: Eq`) for
/// which the "impl_source" must be found. The process of finding an "impl_source" is
/// called "resolving" the `Obligation`. This process consists of
/// either identifying an `impl` (e.g., `impl Eq for i32`) that
/// satisfies the obligation, or else finding a bound that is in
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

impl PredicateObligation<'tcx> {
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

// `PredicateObligation` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(PredicateObligation<'_>, 32);

pub type PredicateObligations<'tcx> = Vec<PredicateObligation<'tcx>>;

pub type Selection<'tcx> = ImplSource<'tcx, PredicateObligation<'tcx>>;

pub struct FulfillmentError<'tcx> {
    pub obligation: PredicateObligation<'tcx>,
    pub code: FulfillmentErrorCode<'tcx>,
    /// Diagnostics only: the 'root' obligation which resulted in
    /// the failure to process `obligation`. This is the obligation
    /// that was initially passed to `register_predicate_obligation`
    pub root_obligation: PredicateObligation<'tcx>,
}

#[derive(Clone)]
pub enum FulfillmentErrorCode<'tcx> {
    CodeSelectionError(SelectionError<'tcx>),
    CodeProjectionError(MismatchedProjectionTypes<'tcx>),
    CodeSubtypeError(ExpectedFound<Ty<'tcx>>, TypeError<'tcx>), // always comes from a SubtypePredicate
    CodeConstEquateError(ExpectedFound<&'tcx Const<'tcx>>, TypeError<'tcx>),
    CodeAmbiguity,
}

pub struct ObligationsDedup<'a, 'tcx, T> {
    obligations: &'a mut Vec<Obligation<'tcx, T>>,
}

impl<'a, 'tcx, T: 'tcx> ObligationsDedup<'a, 'tcx, T>
where
    T: Clone + Hash + Eq,
{
    pub fn from_vec(vec: &'a mut Vec<Obligation<'tcx, T>>) -> Self {
        ObligationsDedup { obligations: vec }
    }

    pub fn extend<'b>(&mut self, iter: impl ExactSizeIterator<Item = Cow<'b, Obligation<'tcx, T>>>)
    where
        'tcx: 'b,
    {
        // obligation tracing has shown that initial batches added to an empty vec do not
        // contain any duplicates, so there's no need to attempt deduplication
        if self.obligations.is_empty() {
            *self.obligations = iter.into_iter().map(Cow::into_owned).collect();
            return;
        }

        let initial_size = self.obligations.len();
        let current_capacity = self.obligations.capacity();
        let iter = iter.into_iter();
        let expected_new = iter.len();
        let combined_size = initial_size + expected_new;

        if combined_size <= 16 || combined_size <= current_capacity {
            // small case/not crossing a power of two. don't bother with dedup
            self.obligations.extend(iter.map(Cow::into_owned));
        } else {
            // crossing power of two threshold. this would incur a vec growth anyway if we didn't do
            // anything. piggyback a dedup on that
            let mut seen = FxHashMap::default();
            seen.reserve(initial_size);

            let mut is_duplicate = move |obligation: &Obligation<'tcx, _>| -> bool {
                return match seen.raw_entry_mut().from_key(obligation) {
                    RawEntryMut::Occupied(..) => true,
                    RawEntryMut::Vacant(vacant) => {
                        vacant.insert(obligation.clone(), ());
                        false
                    }
                };
            };

            self.obligations.retain(|obligation| !is_duplicate(obligation));
            self.obligations.extend(iter.filter_map(|obligation| {
                if is_duplicate(obligation.borrow()) {
                    return None;
                }
                Some(obligation.into_owned())
            }));
        }
    }
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
        root_obligation: PredicateObligation<'tcx>,
    ) -> FulfillmentError<'tcx> {
        FulfillmentError { obligation, code, root_obligation }
    }
}

impl<'tcx> TraitObligation<'tcx> {
    pub fn polarity(&self) -> ty::ImplPolarity {
        self.predicate.skip_binder().polarity
    }

    pub fn self_ty(&self) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.predicate.map_bound(|p| p.self_ty())
    }
}
