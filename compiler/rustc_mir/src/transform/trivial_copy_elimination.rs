//! A simple intra-block copy propagation pass.
//!
//! This pass performs simple forwards-propagation of locals that were assigned within the same MIR
//! block. This is a common pattern introduced by MIR building.

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::{
    MutVisitor, MutatingUseContext, NonMutatingUseContext, NonUseContext, PlaceContext, Visitor,
};
use rustc_middle::mir::{
    Body, Local, Location, Operand, Place, ProjectionElem, Rvalue, Statement, StatementKind,
};
use rustc_middle::ty::TyCtxt;
use smallvec::SmallVec;

use super::MirPass;

const MAX_LOCALS_FOR_PASS: usize = 400;
const MAX_STATEMENTS_PERE_BLOCK: usize = 400;
const AVERAGE_NUM_PATCHES: usize = 2;

pub struct TrivialCopyElimination;

impl<'tcx> MirPass<'tcx> for TrivialCopyElimination {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // if tcx.sess.mir_opt_level() < 2 {
        //     return;
        // }

        let num_locals = body.local_decls.len();
        if num_locals > MAX_LOCALS_FOR_PASS {
            return;
        }
        let mut locals = FxIndexMap::with_capacity_and_hasher(num_locals, Default::default());
        let mut sources = FxIndexMap::with_capacity_and_hasher(num_locals, Default::default());
        let mut patches = FxIndexMap::with_capacity_and_hasher(num_locals, Default::default());
        let mut selected_patches =
            FxIndexMap::with_capacity_and_hasher(num_locals, Default::default());
        let mut patchable_locals = BitSet::new_empty(num_locals);

        for (block, block_data) in body.basic_blocks_mut().iter_enumerated_mut() {
            if block_data.statements.len() > MAX_STATEMENTS_PERE_BLOCK {
                continue;
            }
            let mut visitor = LocalStateCollector {
                locals: &mut locals,
                sources: &mut sources,
                statement_idx: 0,
                place_idx: 0,
                patches: &mut patches,
            };
            visitor.visit_basic_block_data(block, block_data);

            let LocalStateCollector { locals, sources, patches, .. } = visitor;
            // now we need to identify interesting locals and place them into the patches
            for (local, state) in locals.drain(..) {
                let mut patches =
                    if let Some(patches) = patches.remove(&local) { patches } else { continue };
                if patches.is_empty() {
                    continue;
                }

                let skip_last;
                match state {
                    LocalState::Dead => {
                        skip_last = false;
                    }
                    LocalState::CopyIntoLive(.., gen) | LocalState::CopyUsed(.., gen) => {
                        if gen == 0 {
                            continue;
                        }
                        skip_last = true;
                    }
                    LocalState::MovedOut(gen) => {
                        if gen == 0 {
                            continue;
                        }
                        skip_last = false;
                    }
                    _ => continue,
                }
                if skip_last {
                    let _ = patches.pop();
                }
                if patches.is_empty() {
                    continue;
                }
                patchable_locals.insert(local);
                for (src, statement_idx, place_idx) in patches {
                    if selected_patches.contains_key(&(statement_idx, place_idx)) {
                        bug!(
                            "Already contains a patch for statement {} place {}",
                            statement_idx,
                            place_idx
                        );
                    }
                    selected_patches.insert((statement_idx, place_idx), (local, src));
                }
            }
            assert!(locals.is_empty());
            assert!(patches.is_empty());
            sources.clear();

            let mut visitor = Patcher {
                tcx,
                statement_idx: 0,
                place_idx: 0,
                patches: &mut selected_patches,
                patchable_locals: &patchable_locals,
                patched_move_just_yet: false,
            };
            visitor.visit_basic_block_data(block, block_data);

            assert!(visitor.patches.is_empty());
            patchable_locals.clear();
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
enum LocalState<'tcx> {
    NotInteresting,
    Live,
    CopyIntoLive(Option<Place<'tcx>>, u16),
    CopyUsed(Option<Place<'tcx>>, u16),
    MovedOut(u16),
    Dead,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Action<'tcx> {
    CopyFrom(Option<Place<'tcx>>),
    CopyUse,
    MoveUse,
    Invalidate,
    Dead,
}

impl<'tcx> LocalState<'tcx> {
    fn update(self, action: Action<'tcx>) -> Self {
        match (self, action) {
            // trivial case
            (LocalState::Live, Action::Dead) => LocalState::Dead,
            // we copy something in here
            (LocalState::Live, Action::CopyFrom(src)) => LocalState::CopyIntoLive(src, 0),
            // trivial replacement
            (LocalState::CopyIntoLive(.., gen), Action::CopyFrom(src)) => {
                LocalState::CopyIntoLive(src, gen + 1)
            }
            // actual use a freshly copied value
            (LocalState::CopyIntoLive(src, gen), Action::CopyUse) => LocalState::CopyUsed(src, gen),
            (LocalState::CopyIntoLive(.., gen), Action::MoveUse) => LocalState::MovedOut(gen),
            // use already used value
            (LocalState::CopyUsed(src, gen), Action::CopyUse) => LocalState::CopyUsed(src, gen),
            // copy over
            (LocalState::CopyUsed(.., gen) | LocalState::MovedOut(gen), Action::CopyFrom(src)) => {
                LocalState::CopyIntoLive(src, gen + 1)
            }
            // dead when source was invalidated
            (
                LocalState::CopyUsed(..) | LocalState::MovedOut(..) | LocalState::CopyIntoLive(..),
                Action::Dead,
            ) => LocalState::Dead,
            // invalidations of invalid source
            (
                LocalState::CopyIntoLive(.., gen)
                | LocalState::CopyUsed(.., gen)
                | LocalState::MovedOut(.., gen),
                Action::Invalidate,
            ) => LocalState::CopyIntoLive(None, gen),
            _ => LocalState::NotInteresting,
        }
    }
}

/// Determines whether `place` is an assignment source that may later be used instead of the local
/// it is assigned to.
///
/// This is the case only for places that don't dereference pointers (since the dereference
/// operation may not be valid anymore after this point), and don't index a slice (since that uses
/// another local besides the base local, which would need additional tracking).
fn place_eligible(place: &Place<'_>) -> bool {
    place.projection.iter().all(|elem| match elem {
        ProjectionElem::Deref | ProjectionElem::Index(_) => false,

        ProjectionElem::Field(..)
        | ProjectionElem::ConstantIndex { .. }
        | ProjectionElem::Subslice { .. }
        | ProjectionElem::Downcast(..) => true,
    })
}

/// FSM on per-block locals and quasi-locals
/// that allows us to determinee if we want to patch it or not
struct LocalStateCollector<'a, 'tcx> {
    locals: &'a mut FxIndexMap<Local, LocalState<'tcx>>,
    sources: &'a mut FxIndexMap<Local, FxIndexSet<Local>>,
    statement_idx: u16,
    place_idx: u16,
    patches: &'a mut FxIndexMap<Local, SmallVec<[(Place<'tcx>, u16, u16); AVERAGE_NUM_PATCHES]>>,
}

impl<'a, 'tcx> LocalStateCollector<'a, 'tcx> {
    fn log_patch(&mut self, place: &Place<'tcx>) {
        let local = if let Some(local) = place.local_or_deref_local() { local } else { return };
        if let Some(local_state) = self.locals.get(&local) {
            match local_state {
                LocalState::CopyIntoLive(Some(src), ..) | LocalState::CopyUsed(Some(src), ..) => {
                    let entry = self.patches.entry(local).or_default();
                    entry.push((*src, self.statement_idx, self.place_idx));
                }
                _ => {}
            }
        }
    }

    fn progress(&mut self, local: Local, action: Action<'tcx>) {
        if let Some(t) = self.locals.remove(&local) {
            let new = t.update(action);
            self.locals.insert(local, new);
        }

        // if action is a copy into then keep track of sources
        match action {
            Action::CopyFrom(Some(src)) => {
                // peek back into the locals. The generation here is already a new generation
                if let Some(LocalState::CopyIntoLive(..)) = self.locals.get(&local) {
                    // get raw local from the src. If it's eligible it's Some in Action
                    let src_local = src.local;
                    let entries = self.sources.entry(src_local).or_insert(Default::default());
                    entries.insert(local);
                }
            }
            _ => {}
        }
    }

    fn make_live(&mut self, local: Local) {
        if self.locals.contains_key(&local) {
            bug!("We already considered this local as live or quasi-live in our context");
        }
        let new = LocalState::Live;
        self.locals.insert(local, new);
    }

    fn invalidate_source_information(&mut self, local: Local) {
        // if this local is some tracked source
        if let Some(dests) = self.sources.remove(&local) {
            // walk over destinations of this source ordered by generation
            for dest in dests {
                if let Some(state) = self.locals.get(&dest).copied() {
                    // only invalidate if it's for the last generation!
                    match state {
                        LocalState::CopyIntoLive(Some(src), ..) if src.local == local => {
                            self.progress(dest, Action::Invalidate);
                        }
                        LocalState::CopyUsed(Some(src), ..) if src.local == local => {
                            self.progress(dest, Action::Invalidate);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn make_dead(&mut self, local: Local) {
        if let Some(t) = self.locals.remove(&local) {
            let new = t.update(Action::Dead);
            self.locals.insert(local, new);
        }

        // we may need to update the source information too
        // if source is dead we have to check if we currently track this source as a source for one of our
        // locals of interest
        self.invalidate_source_information(local);
    }

    fn invalidate(&mut self, local: Local) {
        if let Some(t) = self.locals.remove(&local) {
            let new = t.update(Action::Invalidate);
            self.locals.insert(local, new);
        }
    }
}

impl<'a, 'tcx: 'a> Visitor<'tcx> for LocalStateCollector<'a, 'tcx> {
    fn visit_local(&mut self, local: &Local, context: PlaceContext, _location: Location) {
        // only edge case of storage live and dead that do not show up in "visit_place"
        match context {
            PlaceContext::NonUse(NonUseContext::StorageLive) => {
                self.make_live(*local);
            }
            PlaceContext::NonUse(NonUseContext::StorageDead) => {
                self.make_dead(*local);
            }
            _ => {}
        }
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        self.super_place(place, context, location);

        let local = place.local;
        let local_is_in_tracked_locals = self.locals.contains_key(&local);
        let local_is_source = self.sources.contains_key(&local);
        // otherwise we need auxilaty information, that will update a generation
        // of disallow patches in a current generation

        match context {
            PlaceContext::NonUse(NonUseContext::StorageLive) => {
                // captured in visit local
                unreachable!("Not visited as `place`");
            }
            PlaceContext::NonUse(NonUseContext::StorageDead) => {
                unreachable!("Not visited as `place`");
            }
            PlaceContext::NonUse(..) => {
                // should be ok
            }
            PlaceContext::MutatingUse(MutatingUseContext::Store) => {
                // we know that is is not a copy into the local,
                // so if it's copy into the source we invalidate
                if local_is_source {
                    self.invalidate_source_information(local);
                }
                if local_is_in_tracked_locals {
                    // do nothing, we have captured this information already
                }
            }
            PlaceContext::MutatingUse(..) => {
                if local_is_source {
                    self.invalidate_source_information(local);
                }
                if local_is_in_tracked_locals {
                    self.invalidate(local);
                }
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) => {
                if local_is_source {
                    // ok
                }
                if local_is_in_tracked_locals {
                    // log first, as we need a current state
                    self.log_patch(place);
                    self.progress(local, Action::CopyUse);
                }
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => {
                if local_is_source {
                    self.invalidate_source_information(local);
                    if local_is_in_tracked_locals {
                        self.invalidate(local);
                    }
                } else if local_is_in_tracked_locals {
                    // log first, as we need a current state
                    self.log_patch(place);
                    self.progress(local, Action::MoveUse);
                }
            }
            // PlaceContext::NonMutatingUse(..) => {
            //     if local_is_source {
            //         // ok
            //     }
            //     if local_is_in_tracked_locals {
            //         progress(self, local, Action::CopyUse);
            //     }
            // },
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect) => {
                if local_is_source {
                    // ok
                }
                if local_is_in_tracked_locals {
                    // log first, as we need a current state
                    self.log_patch(place);
                    self.progress(local, Action::CopyUse);
                }
            }
            PlaceContext::NonMutatingUse(..) => {
                if local_is_source {
                    self.invalidate_source_information(local);
                }
                if local_is_in_tracked_locals {
                    self.invalidate(local);
                }
            }
        }

        self.place_idx += 1;
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            StatementKind::Assign(box (place, rvalue)) => {
                if let Some(local) = place.as_local() {
                    match rvalue {
                        Rvalue::Use(Operand::Copy(src)) => {
                            // we can NOT copy whatever we want into the local
                            let src = if place_eligible(src) { Some(*src) } else { None };
                            if self.locals.get(&local).is_none() {
                                // even if our local is not yet seen due to not
                                // marker "live" in other places,
                                // we can still insert it here is quasi-live state
                                self.make_live(local);
                            }
                            self.progress(local, Action::CopyFrom(src));
                        }
                        _ => {
                            // something else assigned to our local of interest, so
                            // we mark it as copy from non-source, so we bump a generation
                            if self.locals.contains_key(&local) {
                                self.progress(local, Action::CopyFrom(None));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // we visit super after we did capture some information
        self.super_statement(statement, location);

        self.statement_idx += 1;
    }
}

struct Patcher<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    statement_idx: u16,
    place_idx: u16,
    patches: &'a mut FxIndexMap<(u16, u16), (Local, Place<'tcx>)>,
    patchable_locals: &'a BitSet<Local>,
    patched_move_just_yet: bool,
}

impl<'a, 'tcx> Patcher<'a, 'tcx> {
    fn apply_patch(&mut self, place: &mut Place<'tcx>) -> bool {
        if self.patches.is_empty() {
            return false;
        }
        let (local, add_deref) = if let Some(local) = place.as_local() {
            (local, false)
        } else if let Some(local) = place.local_or_deref_local() {
            (local, true)
        } else {
            return false;
        };
        if let Some((local_to_patch, patch_into_place)) =
            self.patches.remove(&(self.statement_idx, self.place_idx))
        {
            assert_eq!(local, local_to_patch);

            let source = if add_deref {
                self.tcx.mk_place_deref(patch_into_place)
            } else {
                patch_into_place
            };
            *place = source;

            return true;
        }

        false
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for Patcher<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        self.super_statement(statement, location);
        self.statement_idx += 1;
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        self.super_operand(operand, location);
        match operand {
            Operand::Move(place) => {
                if self.patched_move_just_yet {
                    *operand = Operand::Copy(*place);
                    self.patched_move_just_yet = false;
                }
            }
            _ => {}
        }
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, _location: Location) {
        if !self.patchable_locals.contains(place.local) {
            self.place_idx += 1;
            return;
        }
        match context {
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Copy) => {
                let _ = self.apply_patch(place);
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Move) => {
                if self.apply_patch(place) {
                    self.patched_move_just_yet = true;
                }
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Inspect) => {
                let _ = self.apply_patch(place);
            }
            _ => {}
        }

        self.place_idx += 1;
    }
}
