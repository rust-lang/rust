//! This file provides API for compiler consumers.

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::LocalDefId;
use rustc_index::IndexVec;
use rustc_middle::bug;
use rustc_middle::mir::{Body, Promoted};
use rustc_middle::ty::TyCtxt;

pub use super::borrow_set::{BorrowData, BorrowSet, TwoPhaseActivation};
pub use super::constraints::OutlivesConstraint;
pub use super::dataflow::{BorrowIndex, Borrows, calculate_borrows_out_of_scope_at_location};
pub use super::place_ext::PlaceExt;
pub use super::places_conflict::{PlaceConflictBias, places_conflict};
pub use super::polonius::legacy::{
    PoloniusFacts as PoloniusInput, PoloniusLocationTable, PoloniusOutput, PoloniusRegionVid,
    RichLocation, RustcFacts,
};
pub use super::region_infer::RegionInferenceContext;
use crate::BorrowCheckRootCtxt;

/// Struct used during mir borrowck to collect bodies with facts for a typeck root and all
/// its nested bodies.
pub(crate) struct BorrowckConsumer<'tcx> {
    options: ConsumerOptions,
    bodies: FxHashMap<LocalDefId, BodyWithBorrowckFacts<'tcx>>,
}

impl<'tcx> BorrowckConsumer<'tcx> {
    pub(crate) fn new(options: ConsumerOptions) -> Self {
        Self { options, bodies: Default::default() }
    }

    pub(crate) fn insert_body(&mut self, def_id: LocalDefId, body: BodyWithBorrowckFacts<'tcx>) {
        if self.bodies.insert(def_id, body).is_some() {
            bug!("unexpected previous body for {def_id:?}");
        }
    }

    /// Should the Polonius input facts be computed?
    pub(crate) fn polonius_input(&self) -> bool {
        matches!(
            self.options,
            ConsumerOptions::PoloniusInputFacts | ConsumerOptions::PoloniusOutputFacts
        )
    }

    /// Should we run Polonius and collect the output facts?
    pub(crate) fn polonius_output(&self) -> bool {
        matches!(self.options, ConsumerOptions::PoloniusOutputFacts)
    }
}

/// Options determining the output behavior of [`get_bodies_with_borrowck_facts`].
///
/// If executing under `-Z polonius` the choice here has no effect, and everything as if
/// [`PoloniusOutputFacts`](ConsumerOptions::PoloniusOutputFacts) had been selected
/// will be retrieved.
#[derive(Debug, Copy, Clone)]
pub enum ConsumerOptions {
    /// Retrieve the [`Body`] along with the [`BorrowSet`]
    /// and [`RegionInferenceContext`]. If you would like the body only, use
    /// [`TyCtxt::mir_promoted`].
    ///
    /// These can be used in conjunction with [`calculate_borrows_out_of_scope_at_location`].
    RegionInferenceContext,
    /// The recommended option. Retrieves the maximal amount of information
    /// without significant slowdowns.
    ///
    /// Implies [`RegionInferenceContext`](ConsumerOptions::RegionInferenceContext),
    /// and additionally retrieve the [`PoloniusLocationTable`] and [`PoloniusInput`] that
    /// would be given to Polonius. Critically, this does not run Polonius, which
    /// one may want to avoid due to performance issues on large bodies.
    PoloniusInputFacts,
    /// Implies [`PoloniusInputFacts`](ConsumerOptions::PoloniusInputFacts),
    /// and additionally runs Polonius to calculate the [`PoloniusOutput`].
    PoloniusOutputFacts,
}

/// A `Body` with information computed by the borrow checker. This struct is
/// intended to be consumed by compiler consumers.
///
/// We need to include the MIR body here because the region identifiers must
/// match the ones in the Polonius facts.
pub struct BodyWithBorrowckFacts<'tcx> {
    /// A mir body that contains region identifiers.
    pub body: Body<'tcx>,
    /// The mir bodies of promoteds.
    pub promoted: IndexVec<Promoted, Body<'tcx>>,
    /// The set of borrows occurring in `body` with data about them.
    pub borrow_set: BorrowSet<'tcx>,
    /// Context generated during borrowck, intended to be passed to
    /// [`calculate_borrows_out_of_scope_at_location`].
    pub region_inference_context: RegionInferenceContext<'tcx>,
    /// The table that maps Polonius points to locations in the table.
    /// Populated when using [`ConsumerOptions::PoloniusInputFacts`]
    /// or [`ConsumerOptions::PoloniusOutputFacts`].
    pub location_table: Option<PoloniusLocationTable>,
    /// Polonius input facts.
    /// Populated when using [`ConsumerOptions::PoloniusInputFacts`]
    /// or [`ConsumerOptions::PoloniusOutputFacts`].
    pub input_facts: Option<Box<PoloniusInput>>,
    /// Polonius output facts. Populated when using
    /// [`ConsumerOptions::PoloniusOutputFacts`].
    pub output_facts: Option<Box<PoloniusOutput>>,
}

/// This function computes borrowck facts for the given def id and all its nested bodies.
/// It must be called with a typeck root which will then borrowck all nested bodies as well.
/// The [`ConsumerOptions`] determine which facts are returned. This function makes a copy
/// of the bodies because it needs to regenerate the region identifiers. It should never be
/// invoked during a typical compilation session due to the unnecessary overhead of
/// returning [`BodyWithBorrowckFacts`].
///
/// Note:
/// *   This function will panic if the required bodies were already stolen. This
///     can, for example, happen when requesting a body of a `const` function
///     because they are evaluated during typechecking. The panic can be avoided
///     by overriding the `mir_borrowck` query. You can find a complete example
///     that shows how to do this at `tests/ui-fulldeps/obtain-borrowck.rs`.
///
/// *   Polonius is highly unstable, so expect regular changes in its signature or other details.
pub fn get_bodies_with_borrowck_facts(
    tcx: TyCtxt<'_>,
    root_def_id: LocalDefId,
    options: ConsumerOptions,
) -> FxHashMap<LocalDefId, BodyWithBorrowckFacts<'_>> {
    let mut root_cx =
        BorrowCheckRootCtxt::new(tcx, root_def_id, Some(BorrowckConsumer::new(options)));
    root_cx.do_mir_borrowck();
    root_cx.consumer.unwrap().bodies
}
