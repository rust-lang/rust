// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::{BorrowSet, BorrowData};

use rustc;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::middle::region;
use rustc::mir::{self, Location, Place, Mir};
use rustc::mir::traversal;
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::ty::{self, Region, TyCtxt};
use rustc::ty::RegionKind;
use rustc::ty::RegionKind::ReScope;
use rustc::util::nodemap::{FxHashMap, FxHashSet};

use rustc_data_structures::bitslice::{BitwiseOperator};
use rustc_data_structures::indexed_set::{IdxSet};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::sync::Lrc;

use dataflow::{BitDenotation, BlockSets, InitialFlow};
pub use dataflow::indexes::{BorrowIndex, ReserveOrActivateIndex};
use borrow_check::nll::region_infer::RegionInferenceContext;
use borrow_check::nll::ToRegionVid;

use syntax_pos::Span;

use std::hash::Hash;
use std::rc::Rc;

/// `Borrows` stores the data used in the analyses that track the flow
/// of borrows.
///
/// It uniquely identifies every borrow (`Rvalue::Ref`) by a
/// `BorrowIndex`, and maps each such index to a `BorrowData`
/// describing the borrow. These indexes are used for representing the
/// borrows in compact bitvectors.
pub struct Borrows<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    scope_tree: Lrc<region::ScopeTree>,
    root_scope: Option<region::Scope>,

    borrow_set: BorrowSet<'tcx>,

    /// NLL region inference context with which NLL queries should be resolved
    nonlexical_regioncx: Option<Rc<RegionInferenceContext<'tcx>>>,
}

impl ReserveOrActivateIndex {
    fn reserved(i: BorrowIndex) -> Self { ReserveOrActivateIndex::new(i.index() * 2) }
    fn active(i: BorrowIndex) -> Self { ReserveOrActivateIndex::new((i.index() * 2) + 1) }

    pub(crate) fn is_reservation(self) -> bool { self.index() % 2 == 0 }
    pub(crate) fn is_activation(self) -> bool { self.index() % 2 == 1}

    pub(crate) fn kind(self) -> &'static str {
        if self.is_reservation() { "reserved" } else { "active" }
    }
    pub(crate) fn borrow_index(self) -> BorrowIndex {
        BorrowIndex::new(self.index() / 2)
    }
}

impl<'a, 'gcx, 'tcx> Borrows<'a, 'gcx, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'gcx, 'tcx>,
               mir: &'a Mir<'tcx>,
               nonlexical_regioncx: Option<Rc<RegionInferenceContext<'tcx>>>,
               def_id: DefId,
               body_id: Option<hir::BodyId>)
               -> Self {
        let scope_tree = tcx.region_scope_tree(def_id);
        let root_scope = body_id.map(|body_id| {
            region::Scope::CallSite(tcx.hir.body(body_id).value.hir_id.local_id)
        });
        let mut visitor = GatherBorrows {
            tcx,
            mir,
            idx_vec: IndexVec::new(),
            location_map: FxHashMap(),
            activation_map: FxHashMap(),
            region_map: FxHashMap(),
            local_map: FxHashMap(),
            region_span_map: FxHashMap(),
            nonlexical_regioncx: nonlexical_regioncx.clone(),
            pending_activations: FxHashMap(),
        };
        for (block, block_data) in traversal::preorder(mir) {
            visitor.visit_basic_block_data(block, block_data);
        }

        // Double check: We should have found an activation for every pending
        // activation.
        assert_eq!(
            visitor
                .pending_activations
                .iter()
                .find(|&(_local, &borrow_index)| {
                    visitor.idx_vec[borrow_index].activation_location.is_none()
                }),
            None,
            "never found an activation for this borrow!",
        );

        return Borrows { tcx: tcx,
                         mir: mir,
                         borrow_set: BorrowSet {
                             borrows: visitor.idx_vec,
                             location_map: visitor.location_map,
                             activation_map: visitor.activation_map,
                             region_map: visitor.region_map,
                             local_map: visitor.local_map,
                             region_span_map: visitor.region_span_map,
                         },
                         scope_tree,
                         root_scope,
                         nonlexical_regioncx };

        struct GatherBorrows<'a, 'gcx: 'tcx, 'tcx: 'a> {
            tcx: TyCtxt<'a, 'gcx, 'tcx>,
            mir: &'a Mir<'tcx>,
            idx_vec: IndexVec<BorrowIndex, BorrowData<'tcx>>,
            location_map: FxHashMap<Location, BorrowIndex>,
            activation_map: FxHashMap<Location, FxHashSet<BorrowIndex>>,
            region_map: FxHashMap<Region<'tcx>, FxHashSet<BorrowIndex>>,
            local_map: FxHashMap<mir::Local, FxHashSet<BorrowIndex>>,
            region_span_map: FxHashMap<RegionKind, Span>,
            nonlexical_regioncx: Option<Rc<RegionInferenceContext<'tcx>>>,

            /// When we encounter a 2-phase borrow statement, it will always
            /// be assigning into a temporary TEMP:
            ///
            ///    TEMP = &foo
            ///
            /// We add TEMP into this map with `b`, where `b` is the index of
            /// the borrow. When we find a later use of this activation, we
            /// remove from the map (and add to the "tombstone" set below).
            pending_activations: FxHashMap<mir::Local, BorrowIndex>,
        }

        impl<'a, 'gcx, 'tcx> Visitor<'tcx> for GatherBorrows<'a, 'gcx, 'tcx> {
            fn visit_assign(&mut self,
                            block: mir::BasicBlock,
                            assigned_place: &mir::Place<'tcx>,
                            rvalue: &mir::Rvalue<'tcx>,
                            location: mir::Location) {
                fn root_local(mut p: &mir::Place<'_>) -> Option<mir::Local> {
                    loop { match p {
                        mir::Place::Projection(pi) => p = &pi.base,
                        mir::Place::Static(_) => return None,
                        mir::Place::Local(l) => return Some(*l)
                    }}
                }

                if let mir::Rvalue::Ref(region, kind, ref borrowed_place) = *rvalue {
                    if is_unsafe_place(self.tcx, self.mir, borrowed_place) { return; }

                    let borrow = BorrowData {
                        kind,
                        region,
                        reserve_location: location,
                        activation_location: None,
                        borrowed_place: borrowed_place.clone(),
                        assigned_place: assigned_place.clone(),
                    };
                    let idx = self.idx_vec.push(borrow);
                    self.location_map.insert(location, idx);

                    self.insert_as_pending_if_two_phase(
                        location,
                        &assigned_place,
                        region,
                        kind,
                        idx,
                    );

                    insert(&mut self.region_map, &region, idx);
                    if let Some(local) = root_local(borrowed_place) {
                        insert(&mut self.local_map, &local, idx);
                    }
                }

                return self.super_assign(block, assigned_place, rvalue, location);

                fn insert<'a, K, V>(map: &'a mut FxHashMap<K, FxHashSet<V>>,
                                    k: &K,
                                    v: V)
                    where K: Clone+Eq+Hash, V: Eq+Hash
                {
                    map.entry(k.clone())
                        .or_insert(FxHashSet())
                        .insert(v);
                }
            }

            fn visit_place(
                &mut self,
                place: &mir::Place<'tcx>,
                context: PlaceContext<'tcx>,
                location: Location,
            ) {
                self.super_place(place, context, location);

                // We found a use of some temporary TEMP...
                if let Place::Local(temp) = place {
                    // ... check whether we (earlier) saw a 2-phase borrow like
                    //
                    //     TMP = &mut place
                    match self.pending_activations.get(temp) {
                        Some(&borrow_index) => {
                            let borrow_data = &mut self.idx_vec[borrow_index];

                            // Watch out: the use of TMP in the borrow
                            // itself doesn't count as an
                            // activation. =)
                            if borrow_data.reserve_location == location
                                && context == PlaceContext::Store
                            {
                                return;
                            }

                            if let Some(other_activation) = borrow_data.activation_location {
                                span_bug!(
                                    self.mir.source_info(location).span,
                                    "found two activations for 2-phase borrow temporary {:?}: \
                                     {:?} and {:?}",
                                    temp,
                                    location,
                                    other_activation,
                                );
                            }

                            // Otherwise, this is the unique later use
                            // that we expect.
                            borrow_data.activation_location = Some(location);
                            self.activation_map
                                .entry(location)
                                .or_insert(FxHashSet())
                                .insert(borrow_index);
                        }

                        None => {}
                    }
                }
            }

            fn visit_rvalue(&mut self,
                            rvalue: &mir::Rvalue<'tcx>,
                            location: mir::Location) {
                if let mir::Rvalue::Ref(region, kind, ref place) = *rvalue {
                    // double-check that we already registered a BorrowData for this

                    let borrow_index = self.location_map[&location];
                    let borrow_data = &self.idx_vec[borrow_index];
                    assert_eq!(borrow_data.reserve_location, location);
                    assert_eq!(borrow_data.kind, kind);
                    assert_eq!(borrow_data.region, region);
                    assert_eq!(borrow_data.borrowed_place, *place);
                }

                return self.super_rvalue(rvalue, location);
            }

            fn visit_statement(&mut self,
                               block: mir::BasicBlock,
                               statement: &mir::Statement<'tcx>,
                               location: Location) {
                if let mir::StatementKind::EndRegion(region_scope) = statement.kind {
                    self.region_span_map.insert(ReScope(region_scope), statement.source_info.span);
                }
                return self.super_statement(block, statement, location);
            }
        }

        /// A MIR visitor that determines if a specific place is used in a two-phase activating
        /// manner in a given chunk of MIR.
        struct ContainsUseOfPlace<'b, 'tcx: 'b> {
            target: &'b Place<'tcx>,
            use_found: bool,
        }

        impl<'b, 'tcx: 'b> ContainsUseOfPlace<'b, 'tcx> {
            fn new(place: &'b Place<'tcx>) -> Self {
                Self { target: place, use_found: false }
            }

            /// return whether `context` should be considered a "use" of a
            /// place found in that context. "Uses" activate associated
            /// borrows (at least when such uses occur while the borrow also
            /// has a reservation at the time).
            fn is_potential_use(context: PlaceContext) -> bool {
                match context {
                    // storage effects on a place do not activate it
                    PlaceContext::StorageLive | PlaceContext::StorageDead => false,

                    // validation effects do not activate a place
                    //
                    // FIXME: Should they? Is it just another read? Or can we
                    // guarantee it won't dereference the stored address? How
                    // "deep" does validation go?
                    PlaceContext::Validate => false,

                    // FIXME: This is here to not change behaviour from before
                    // AsmOutput existed, but it's not necessarily a pure overwrite.
                    // so it's possible this should activate the place.
                    PlaceContext::AsmOutput |
                    // pure overwrites of a place do not activate it. (note
                    // PlaceContext::Call is solely about dest place)
                    PlaceContext::Store | PlaceContext::Call => false,

                    // reads of a place *do* activate it
                    PlaceContext::Move |
                    PlaceContext::Copy |
                    PlaceContext::Drop |
                    PlaceContext::Inspect |
                    PlaceContext::Borrow { .. } |
                    PlaceContext::Projection(..) => true,
                }
            }
        }

        impl<'b, 'tcx: 'b> Visitor<'tcx> for ContainsUseOfPlace<'b, 'tcx> {
            fn visit_place(&mut self,
                           place: &mir::Place<'tcx>,
                           context: PlaceContext<'tcx>,
                           location: Location) {
                if Self::is_potential_use(context) && place == self.target {
                    self.use_found = true;
                    return;
                    // There is no need to keep checking the statement, we already found a use
                }

                self.super_place(place, context, location);
            }
        }

        impl<'a, 'gcx, 'tcx> GatherBorrows<'a, 'gcx, 'tcx> {
            /// Returns true if the borrow represented by `kind` is
            /// allowed to be split into separate Reservation and
            /// Activation phases.
            fn allow_two_phase_borrow(&self, kind: mir::BorrowKind) -> bool {
                self.tcx.two_phase_borrows() &&
                    (kind.allows_two_phase_borrow() ||
                     self.tcx.sess.opts.debugging_opts.two_phase_beyond_autoref)
            }

            /// Returns true if the given location contains an NLL-activating use of the given place
            fn location_contains_use(&self, location: Location, place: &Place) -> bool {
                let mut use_checker = ContainsUseOfPlace::new(place);
                let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
                    panic!("could not find block at location {:?}", location);
                });
                if location.statement_index != block.statements.len() {
                    // This is a statement
                    let stmt = block.statements.get(location.statement_index).unwrap_or_else(|| {
                        panic!("could not find statement at location {:?}");
                    });
                    use_checker.visit_statement(location.block, stmt, location);
                } else {
                    // This is a terminator
                    match block.terminator {
                        Some(ref term) => {
                            use_checker.visit_terminator(location.block, term, location);
                        }
                        None => {
                            // There is no way for Place to be used by the terminator if there is no
                            // terminator
                        }
                    }
                }

                use_checker.use_found
            }

            /// Determines if the provided region is terminated after the provided location.
            /// EndRegion statements terminate their enclosed region::Scope.
            /// We also consult with the NLL region inference engine, should one be available
            fn region_terminated_after(&self, region: Region<'tcx>, location: Location) -> bool {
                let block_data = &self.mir[location.block];
                if location.statement_index != block_data.statements.len() {
                    let stmt = &block_data.statements[location.statement_index];
                    if let mir::StatementKind::EndRegion(region_scope) = stmt.kind {
                        if &ReScope(region_scope) == region {
                            // We encountered an EndRegion statement that terminates the provided
                            // region
                            return true;
                        }
                    }
                }
                if let Some(ref regioncx) = self.nonlexical_regioncx {
                    if !regioncx.region_contains_point(region, location) {
                        // NLL says the region has ended already
                        return true;
                    }
                }

                false
            }

            /// If this is a two-phase borrow, then we will record it
            /// as "pending" until we find the activating use.
            fn insert_as_pending_if_two_phase(
                &mut self,
                start_location: Location,
                assigned_place: &mir::Place<'tcx>,
                region: Region<'tcx>,
                kind: mir::BorrowKind,
                borrow_index: BorrowIndex,
            ) {
                debug!(
                    "Borrows::insert_as_pending_if_two_phase({:?}, {:?}, {:?}, {:?})",
                    start_location, assigned_place, region, borrow_index,
                );

                if !self.allow_two_phase_borrow(kind) {
                    debug!("  -> {:?}", start_location);
                    return;
                }

                // When we encounter a 2-phase borrow statement, it will always
                // be assigning into a temporary TEMP:
                //
                //    TEMP = &foo
                //
                // so extract `temp`.
                let temp = if let &mir::Place::Local(temp) = assigned_place {
                    temp
                } else {
                    span_bug!(
                        self.mir.source_info(start_location).span,
                        "expected 2-phase borrow to assign to a local, not `{:?}`",
                        assigned_place,
                    );
                };

                // Insert `temp` into the list of pending activations. From
                // now on, we'll be on the lookout for a use of it. Note that
                // we are guaranteed that this use will come after the
                // assignment.
                let old_value = self.pending_activations.insert(temp, borrow_index);
                assert!(old_value.is_none());
            }
        }
    }

    /// Returns the span for the "end point" given region. This will
    /// return `None` if NLL is enabled, since that concept has no
    /// meaning there.  Otherwise, return region span if it exists and
    /// span for end of the function if it doesn't exist.
    pub(crate) fn opt_region_end_span(&self, region: &Region) -> Option<Span> {
        match self.nonlexical_regioncx {
            Some(_) => None,
            None => {
                match self.borrow_set.region_span_map.get(region) {
                    Some(span) => Some(self.tcx.sess.codemap().end_point(*span)),
                    None => Some(self.tcx.sess.codemap().end_point(self.mir.span))
                }
            }
        }
    }

    crate fn borrows(&self) -> &IndexVec<BorrowIndex, BorrowData<'tcx>> { &self.borrow_set.borrows }

    pub fn scope_tree(&self) -> &Lrc<region::ScopeTree> { &self.scope_tree }

    pub fn location(&self, idx: BorrowIndex) -> &Location {
        &self.borrow_set.borrows[idx].reserve_location
    }

    /// Add all borrows to the kill set, if those borrows are out of scope at `location`.
    /// That means either they went out of either a nonlexical scope, if we care about those
    /// at the moment, or the location represents a lexical EndRegion
    fn kill_loans_out_of_scope_at_location(&self,
                                           sets: &mut BlockSets<ReserveOrActivateIndex>,
                                           location: Location) {
        if let Some(ref regioncx) = self.nonlexical_regioncx {
            // NOTE: The state associated with a given `location`
            // reflects the dataflow on entry to the statement. If it
            // does not contain `borrow_region`, then then that means
            // that the statement at `location` kills the borrow.
            //
            // We are careful always to call this function *before* we
            // set up the gen-bits for the statement or
            // termanator. That way, if the effect of the statement or
            // terminator *does* introduce a new loan of the same
            // region, then setting that gen-bit will override any
            // potential kill introduced here.
            for (borrow_index, borrow_data) in self.borrow_set.borrows.iter_enumerated() {
                let borrow_region = borrow_data.region.to_region_vid();
                if !regioncx.region_contains_point(borrow_region, location) {
                    sets.kill(&ReserveOrActivateIndex::reserved(borrow_index));
                    sets.kill(&ReserveOrActivateIndex::active(borrow_index));
                }
            }
        }
    }

    fn kill_borrows_on_local(&self,
                             sets: &mut BlockSets<ReserveOrActivateIndex>,
                             local: &rustc::mir::Local)
    {
        if let Some(borrow_indexes) = self.borrow_set.local_map.get(local) {
            sets.kill_all(borrow_indexes.iter()
                          .map(|b| ReserveOrActivateIndex::reserved(*b)));
            sets.kill_all(borrow_indexes.iter()
                          .map(|b| ReserveOrActivateIndex::active(*b)));
        }
    }

    /// Performs the activations for a given location
    fn perform_activations_at_location(&self,
                                       sets: &mut BlockSets<ReserveOrActivateIndex>,
                                       location: Location) {
        // Handle activations
        match self.borrow_set.activation_map.get(&location) {
            Some(activations) => {
                for activated in activations {
                    debug!("activating borrow {:?}", activated);
                    sets.gen(&ReserveOrActivateIndex::active(*activated))
                }
            }
            None => {}
        }
    }
}

impl<'a, 'gcx, 'tcx> BitDenotation for Borrows<'a, 'gcx, 'tcx> {
    type Idx = ReserveOrActivateIndex;
    fn name() -> &'static str { "borrows" }
    fn bits_per_block(&self) -> usize {
        self.borrow_set.borrows.len() * 2
    }

    fn start_block_effect(&self, _entry_set: &mut IdxSet<ReserveOrActivateIndex>) {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect on
        // `_sets`.
    }

    fn before_statement_effect(&self,
                               sets: &mut BlockSets<ReserveOrActivateIndex>,
                               location: Location) {
        debug!("Borrows::before_statement_effect sets: {:?} location: {:?}", sets, location);
        self.kill_loans_out_of_scope_at_location(sets, location);
    }

    fn statement_effect(&self, sets: &mut BlockSets<ReserveOrActivateIndex>, location: Location) {
        debug!("Borrows::statement_effect sets: {:?} location: {:?}", sets, location);

        let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });
        let stmt = block.statements.get(location.statement_index).unwrap_or_else(|| {
            panic!("could not find statement at location {:?}");
        });

        self.perform_activations_at_location(sets, location);
        self.kill_loans_out_of_scope_at_location(sets, location);

        match stmt.kind {
            // EndRegion kills any borrows (reservations and active borrows both)
            mir::StatementKind::EndRegion(region_scope) => {
                if let Some(borrow_indexes) =
                    self.borrow_set.region_map.get(&ReScope(region_scope))
                {
                    assert!(self.nonlexical_regioncx.is_none());
                    for idx in borrow_indexes {
                        sets.kill(&ReserveOrActivateIndex::reserved(*idx));
                        sets.kill(&ReserveOrActivateIndex::active(*idx));
                    }
                } else {
                    // (if there is no entry, then there are no borrows to be tracked)
                }
            }

            mir::StatementKind::Assign(ref lhs, ref rhs) => {
                // Make sure there are no remaining borrows for variables
                // that are assigned over.
                if let Place::Local(ref local) = *lhs {
                    // FIXME: Handle the case in which we're assigning over
                    // a projection (`foo.bar`).
                    self.kill_borrows_on_local(sets, local);
                }

                // NOTE: if/when the Assign case is revised to inspect
                // the assigned_place here, make sure to also
                // re-consider the current implementations of the
                // propagate_call_return method.

                if let mir::Rvalue::Ref(region, _, ref place) = *rhs {
                    if is_unsafe_place(self.tcx, self.mir, place) { return; }
                    let index = self.borrow_set.location_map.get(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {:?}", location);
                    });

                    if let RegionKind::ReEmpty = region {
                        // If the borrowed value dies before the borrow is used, the region for
                        // the borrow can be empty. Don't track the borrow in that case.
                        sets.kill(&ReserveOrActivateIndex::active(*index));
                        return
                    }

                    assert!(self.borrow_set.region_map.get(region).unwrap_or_else(|| {
                        panic!("could not find BorrowIndexs for region {:?}", region);
                    }).contains(&index));
                    sets.gen(&ReserveOrActivateIndex::reserved(*index));

                    // Issue #46746: Two-phase borrows handles
                    // stmts of form `Tmp = &mut Borrow` ...
                    match lhs {
                        Place::Local(..) | Place::Static(..) => {} // okay
                        Place::Projection(..) => {
                            // ... can assign into projections,
                            // e.g. `box (&mut _)`. Current
                            // conservative solution: force
                            // immediate activation here.
                            sets.gen(&ReserveOrActivateIndex::active(*index));
                        }
                    }
                }
            }

            mir::StatementKind::StorageDead(local) => {
                // Make sure there are no remaining borrows for locals that
                // are gone out of scope.
                self.kill_borrows_on_local(sets, &local)
            }

            mir::StatementKind::InlineAsm { ref outputs, ref asm, .. } => {
                for (output, kind) in outputs.iter().zip(&asm.outputs) {
                    if !kind.is_indirect && !kind.is_rw {
                        // Make sure there are no remaining borrows for direct
                        // output variables.
                        if let Place::Local(ref local) = *output {
                            // FIXME: Handle the case in which we're assigning over
                            // a projection (`foo.bar`).
                            self.kill_borrows_on_local(sets, local);
                        }
                    }
                }
            }

            mir::StatementKind::SetDiscriminant { .. } |
            mir::StatementKind::StorageLive(..) |
            mir::StatementKind::Validate(..) |
            mir::StatementKind::UserAssertTy(..) |
            mir::StatementKind::Nop => {}

        }
    }

    fn before_terminator_effect(&self,
                                sets: &mut BlockSets<ReserveOrActivateIndex>,
                                location: Location) {
        debug!("Borrows::before_terminator_effect sets: {:?} location: {:?}", sets, location);
        self.kill_loans_out_of_scope_at_location(sets, location);
    }

    fn terminator_effect(&self, sets: &mut BlockSets<ReserveOrActivateIndex>, location: Location) {
        debug!("Borrows::terminator_effect sets: {:?} location: {:?}", sets, location);

        let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });

        let term = block.terminator();
        self.perform_activations_at_location(sets, location);
        self.kill_loans_out_of_scope_at_location(sets, location);


        match term.kind {
            mir::TerminatorKind::Resume |
            mir::TerminatorKind::Return |
            mir::TerminatorKind::GeneratorDrop => {
                // When we return from the function, then all `ReScope`-style regions
                // are guaranteed to have ended.
                // Normally, there would be `EndRegion` statements that come before,
                // and hence most of these loans will already be dead -- but, in some cases
                // like unwind paths, we do not always emit `EndRegion` statements, so we
                // add some kills here as a "backup" and to avoid spurious error messages.
                for (borrow_index, borrow_data) in self.borrow_set.borrows.iter_enumerated() {
                    if let ReScope(scope) = borrow_data.region {
                        // Check that the scope is not actually a scope from a function that is
                        // a parent of our closure. Note that the CallSite scope itself is
                        // *outside* of the closure, for some weird reason.
                        if let Some(root_scope) = self.root_scope {
                            if *scope != root_scope &&
                                self.scope_tree.is_subscope_of(*scope, root_scope)
                            {
                                sets.kill(&ReserveOrActivateIndex::reserved(borrow_index));
                                sets.kill(&ReserveOrActivateIndex::active(borrow_index));
                            }
                        }
                    }
                }
            }
            mir::TerminatorKind::Abort |
            mir::TerminatorKind::SwitchInt {..} |
            mir::TerminatorKind::Drop {..} |
            mir::TerminatorKind::DropAndReplace {..} |
            mir::TerminatorKind::Call {..} |
            mir::TerminatorKind::Assert {..} |
            mir::TerminatorKind::Yield {..} |
            mir::TerminatorKind::Goto {..} |
            mir::TerminatorKind::FalseEdges {..} |
            mir::TerminatorKind::FalseUnwind {..} |
            mir::TerminatorKind::Unreachable => {}
        }
    }

    fn propagate_call_return(&self,
                             _in_out: &mut IdxSet<ReserveOrActivateIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             _dest_place: &mir::Place) {
        // there are no effects on borrows from method call return...
        //
        // ... but if overwriting a place can affect flow state, then
        // latter is not true; see NOTE on Assign case in
        // statement_effect_on_borrows.
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for Borrows<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // union effects of preds when computing reservations
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for Borrows<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = nothing is reserved or activated yet
    }
}

fn is_unsafe_place<'a, 'gcx: 'tcx, 'tcx: 'a>(
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    place: &mir::Place<'tcx>
) -> bool {
    use self::mir::Place::*;
    use self::mir::ProjectionElem;

    match *place {
        Local(_) => false,
        Static(ref static_) => tcx.is_static(static_.def_id) == Some(hir::Mutability::MutMutable),
        Projection(ref proj) => {
            match proj.elem {
                ProjectionElem::Field(..) |
                ProjectionElem::Downcast(..) |
                ProjectionElem::Subslice { .. } |
                ProjectionElem::ConstantIndex { .. } |
                ProjectionElem::Index(_) => {
                    is_unsafe_place(tcx, mir, &proj.base)
                }
                ProjectionElem::Deref => {
                    let ty = proj.base.ty(mir, tcx).to_ty(tcx);
                    match ty.sty {
                        ty::TyRawPtr(..) => true,
                        _ => is_unsafe_place(tcx, mir, &proj.base),
                    }
                }
            }
        }
    }
}
