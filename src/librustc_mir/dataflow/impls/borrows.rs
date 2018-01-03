// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::middle::region;
use rustc::mir::{self, Location, Place, Mir};
use rustc::mir::visit::{PlaceContext, Visitor};
use rustc::ty::{self, Region, TyCtxt};
use rustc::ty::RegionKind;
use rustc::ty::RegionKind::ReScope;
use rustc::util::nodemap::{FxHashMap, FxHashSet};

use rustc_data_structures::bitslice::{BitwiseOperator};
use rustc_data_structures::indexed_set::{IdxSet};
use rustc_data_structures::indexed_vec::{Idx, IndexVec};

use dataflow::{BitDenotation, BlockSets, InitialFlow};
pub use dataflow::indexes::{BorrowIndex, ReserveOrActivateIndex};
use borrow_check::nll::region_infer::RegionInferenceContext;
use borrow_check::nll::ToRegionVid;

use syntax_pos::Span;

use std::fmt;
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
    scope_tree: Rc<region::ScopeTree>,
    root_scope: Option<region::Scope>,

    /// The fundamental map relating bitvector indexes to the borrows
    /// in the MIR.
    borrows: IndexVec<BorrowIndex, BorrowData<'tcx>>,

    /// Each borrow is also uniquely identified in the MIR by the
    /// `Location` of the assignment statement in which it appears on
    /// the right hand side; we map each such location to the
    /// corresponding `BorrowIndex`.
    location_map: FxHashMap<Location, BorrowIndex>,

    /// Every borrow in MIR is immediately stored into a place via an
    /// assignment statement. This maps each such assigned place back
    /// to its borrow-indexes.
    assigned_map: FxHashMap<Place<'tcx>, FxHashSet<BorrowIndex>>,

    /// Every borrow has a region; this maps each such regions back to
    /// its borrow-indexes.
    region_map: FxHashMap<Region<'tcx>, FxHashSet<BorrowIndex>>,
    local_map: FxHashMap<mir::Local, FxHashSet<BorrowIndex>>,
    region_span_map: FxHashMap<RegionKind, Span>,
    nonlexical_regioncx: Option<Rc<RegionInferenceContext<'tcx>>>,
}

// Two-phase borrows actually requires two flow analyses; they need
// to be separate because the final results of the first are used to
// construct the gen+kill sets for the second. (The dataflow system
// is not designed to allow the gen/kill sets to change during the
// fixed-point iteration.)

/// The `Reservations` analysis is the first of the two flow analyses
/// tracking (phased) borrows. It computes where a borrow is reserved;
/// i.e. where it can reach in the control flow starting from its
/// initial `assigned = &'rgn borrowed` statement, and ending
/// whereever `'rgn` itself ends.
pub(crate) struct Reservations<'a, 'gcx: 'tcx, 'tcx: 'a>(pub(crate) Borrows<'a, 'gcx, 'tcx>);

/// The `ActiveBorrows` analysis is the second of the two flow
/// analyses tracking (phased) borrows. It computes where any given
/// borrow `&assigned = &'rgn borrowed` is *active*, which starts at
/// the first use of `assigned` after the reservation has started, and
/// ends whereever `'rgn` itself ends.
pub(crate) struct ActiveBorrows<'a, 'gcx: 'tcx, 'tcx: 'a>(pub(crate) Borrows<'a, 'gcx, 'tcx>);

impl<'a, 'gcx, 'tcx> Reservations<'a, 'gcx, 'tcx> {
    pub(crate) fn new(b: Borrows<'a, 'gcx, 'tcx>) -> Self { Reservations(b) }
    pub(crate) fn location(&self, idx: ReserveOrActivateIndex) -> &Location {
        self.0.location(idx.borrow_index())
    }
}

impl<'a, 'gcx, 'tcx> ActiveBorrows<'a, 'gcx, 'tcx> {
    pub(crate) fn new(r: Reservations<'a, 'gcx, 'tcx>) -> Self { ActiveBorrows(r.0) }
    pub(crate) fn location(&self, idx: ReserveOrActivateIndex) -> &Location {
        self.0.location(idx.borrow_index())
    }
}

// temporarily allow some dead fields: `kind` and `region` will be
// needed by borrowck; `borrowed_place` will probably be a MovePathIndex when
// that is extended to include borrowed data paths.
#[allow(dead_code)]
#[derive(Debug)]
pub struct BorrowData<'tcx> {
    pub(crate) location: Location,
    pub(crate) kind: mir::BorrowKind,
    pub(crate) region: Region<'tcx>,
    pub(crate) borrowed_place: mir::Place<'tcx>,
    pub(crate) assigned_place: mir::Place<'tcx>,
}

impl<'tcx> fmt::Display for BorrowData<'tcx> {
    fn fmt(&self, w: &mut fmt::Formatter) -> fmt::Result {
        let kind = match self.kind {
            mir::BorrowKind::Shared => "",
            mir::BorrowKind::Unique => "uniq ",
            mir::BorrowKind::Mut => "mut ",
        };
        let region = format!("{}", self.region);
        let region = if region.len() > 0 { format!("{} ", region) } else { region };
        write!(w, "&{}{}{:?}", region, kind, self.borrowed_place)
    }
}

impl ReserveOrActivateIndex {
    fn reserved(i: BorrowIndex) -> Self { ReserveOrActivateIndex::new((i.index() * 2)) }
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
            assigned_map: FxHashMap(),
            region_map: FxHashMap(),
            local_map: FxHashMap(),
            region_span_map: FxHashMap()
        };
        visitor.visit_mir(mir);
        return Borrows { tcx: tcx,
                         mir: mir,
                         borrows: visitor.idx_vec,
                         scope_tree,
                         root_scope,
                         location_map: visitor.location_map,
                         assigned_map: visitor.assigned_map,
                         region_map: visitor.region_map,
                         local_map: visitor.local_map,
                         region_span_map: visitor.region_span_map,
                         nonlexical_regioncx };

        struct GatherBorrows<'a, 'gcx: 'tcx, 'tcx: 'a> {
            tcx: TyCtxt<'a, 'gcx, 'tcx>,
            mir: &'a Mir<'tcx>,
            idx_vec: IndexVec<BorrowIndex, BorrowData<'tcx>>,
            location_map: FxHashMap<Location, BorrowIndex>,
            assigned_map: FxHashMap<Place<'tcx>, FxHashSet<BorrowIndex>>,
            region_map: FxHashMap<Region<'tcx>, FxHashSet<BorrowIndex>>,
            local_map: FxHashMap<mir::Local, FxHashSet<BorrowIndex>>,
            region_span_map: FxHashMap<RegionKind, Span>,
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
                        location, kind, region,
                        borrowed_place: borrowed_place.clone(),
                        assigned_place: assigned_place.clone(),
                    };
                    let idx = self.idx_vec.push(borrow);
                    self.location_map.insert(location, idx);

                    insert(&mut self.assigned_map, assigned_place, idx);
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

            fn visit_rvalue(&mut self,
                            rvalue: &mir::Rvalue<'tcx>,
                            location: mir::Location) {
                if let mir::Rvalue::Ref(region, kind, ref place) = *rvalue {
                    // double-check that we already registered a BorrowData for this

                    let mut found_it = false;
                    for idx in &self.region_map[region] {
                        let bd = &self.idx_vec[*idx];
                        if bd.location == location &&
                            bd.kind == kind &&
                            bd.region == region &&
                            bd.borrowed_place == *place
                        {
                            found_it = true;
                            break;
                        }
                    }
                    assert!(found_it, "Ref {:?} at {:?} missing BorrowData", rvalue, location);
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
    }

    pub fn borrows(&self) -> &IndexVec<BorrowIndex, BorrowData<'tcx>> { &self.borrows }

    pub fn scope_tree(&self) -> &Rc<region::ScopeTree> { &self.scope_tree }

    pub fn location(&self, idx: BorrowIndex) -> &Location {
        &self.borrows[idx].location
    }

    /// Add all borrows to the kill set, if those borrows are out of scope at `location`.
    ///
    /// `is_activations` tracks whether we are in the Reservations or
    /// the ActiveBorrows flow analysis, and does not set the
    /// activation kill bits in the former case. (Technically, we
    /// could set those kill bits without such a guard, since they are
    /// never gen'ed by Reservations in the first place.  But it makes
    /// the instrumentation and graph renderings nicer to leave
    /// activations out when of the Reservations kill sets.)
    fn kill_loans_out_of_scope_at_location(&self,
                                           sets: &mut BlockSets<ReserveOrActivateIndex>,
                                           location: Location,
                                           is_activations: bool) {
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
            for (borrow_index, borrow_data) in self.borrows.iter_enumerated() {
                let borrow_region = borrow_data.region.to_region_vid();
                if !regioncx.region_contains_point(borrow_region, location) {
                    sets.kill(&ReserveOrActivateIndex::reserved(borrow_index));
                    if is_activations {
                        sets.kill(&ReserveOrActivateIndex::active(borrow_index));
                    }
                }
            }
        }
    }

    /// Models statement effect in Reservations and ActiveBorrows flow
    /// analyses; `is activations` tells us if we are in the latter
    /// case.
    fn statement_effect_on_borrows(&self,
                                   sets: &mut BlockSets<ReserveOrActivateIndex>,
                                   location: Location,
                                   is_activations: bool) {
        let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });
        let stmt = block.statements.get(location.statement_index).unwrap_or_else(|| {
            panic!("could not find statement at location {:?}");
        });

        // Do kills introduced by NLL before setting up any potential
        // gens. (See NOTE in kill_loans_out_of_scope_at_location.)
        self.kill_loans_out_of_scope_at_location(sets, location, is_activations);

        if is_activations {
            // INVARIANT: `sets.on_entry` accurately captures
            // reservations on entry to statement (b/c
            // accumulates_intrablock_state is overridden for
            // ActiveBorrows).
            //
            // Now compute the activations generated by uses within
            // the statement based on that reservation state.
            let mut find = FindPlaceUses { sets, assigned_map: &self.assigned_map };
            find.visit_statement(location.block, stmt, location);
        }

        match stmt.kind {
            // EndRegion kills any borrows (reservations and active borrows both)
            mir::StatementKind::EndRegion(region_scope) => {
                if let Some(borrow_indexes) = self.region_map.get(&ReScope(region_scope)) {
                    assert!(self.nonlexical_regioncx.is_none());
                    for idx in borrow_indexes {
                        sets.kill(&ReserveOrActivateIndex::reserved(*idx));
                        if is_activations {
                            sets.kill(&ReserveOrActivateIndex::active(*idx));
                        }
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
                    self.kill_borrows_on_local(sets, local, is_activations);
                }

                // NOTE: if/when the Assign case is revised to inspect
                // the assigned_place here, make sure to also
                // re-consider the current implementations of the
                // propagate_call_return method.

                if let mir::Rvalue::Ref(region, _, ref place) = *rhs {
                    if is_unsafe_place(self.tcx, self.mir, place) { return; }
                    if let RegionKind::ReEmpty = region {
                        // If the borrowed value is dead, the region for it
                        // can be empty. Don't track the borrow in that case.
                        return
                    }

                    let index = self.location_map.get(&location).unwrap_or_else(|| {
                        panic!("could not find BorrowIndex for location {:?}", location);
                    });
                    assert!(self.region_map.get(region).unwrap_or_else(|| {
                        panic!("could not find BorrowIndexs for region {:?}", region);
                    }).contains(&index));
                    sets.gen(&ReserveOrActivateIndex::reserved(*index));

                    if is_activations {
                        // Issue #46746: Two-phase borrows handles
                        // stmts of form `Tmp = &mut Borrow` ...
                        match lhs {
                            Place::Local(..) => {} // okay
                            Place::Static(..) => unreachable!(), // (filtered by is_unsafe_place)
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
            }

            mir::StatementKind::StorageDead(local) => {
                // Make sure there are no remaining borrows for locals that
                // are gone out of scope.
                self.kill_borrows_on_local(sets, &local, is_activations)
            }

            mir::StatementKind::InlineAsm { ref outputs, ref asm, .. } => {
                for (output, kind) in outputs.iter().zip(&asm.outputs) {
                    if !kind.is_indirect && !kind.is_rw {
                        // Make sure there are no remaining borrows for direct
                        // output variables.
                        if let Place::Local(ref local) = *output {
                            // FIXME: Handle the case in which we're assigning over
                            // a projection (`foo.bar`).
                            self.kill_borrows_on_local(sets, local, is_activations);
                        }
                    }
                }
            }

            mir::StatementKind::SetDiscriminant { .. } |
            mir::StatementKind::StorageLive(..) |
            mir::StatementKind::Validate(..) |
            mir::StatementKind::Nop => {}

        }
    }

    fn kill_borrows_on_local(&self,
                             sets: &mut BlockSets<ReserveOrActivateIndex>,
                             local: &rustc::mir::Local,
                             is_activations: bool)
    {
        if let Some(borrow_indexes) = self.local_map.get(local) {
            sets.kill_all(borrow_indexes.iter()
                            .map(|b| ReserveOrActivateIndex::reserved(*b)));
            if is_activations {
                sets.kill_all(borrow_indexes.iter()
                                .map(|b| ReserveOrActivateIndex::active(*b)));
            }
        }
    }

    /// Models terminator effect in Reservations and ActiveBorrows
    /// flow analyses; `is activations` tells us if we are in the
    /// latter case.
    fn terminator_effect_on_borrows(&self,
                                    sets: &mut BlockSets<ReserveOrActivateIndex>,
                                    location: Location,
                                    is_activations: bool) {
        let block = &self.mir.basic_blocks().get(location.block).unwrap_or_else(|| {
            panic!("could not find block at location {:?}", location);
        });

        // Do kills introduced by NLL before setting up any potential
        // gens. (See NOTE in kill_loans_out_of_scope_at_location.)
        self.kill_loans_out_of_scope_at_location(sets, location, is_activations);

        let term = block.terminator();
        if is_activations {
            // INVARIANT: `sets.on_entry` accurately captures
            // reservations on entry to terminator (b/c
            // accumulates_intrablock_state is overridden for
            // ActiveBorrows).
            //
            // Now compute effect of the terminator on the activations
            // themselves in the ActiveBorrows state.
            let mut find = FindPlaceUses { sets, assigned_map: &self.assigned_map };
            find.visit_terminator(location.block, term, location);
        }

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
                for (borrow_index, borrow_data) in self.borrows.iter_enumerated() {
                    if let ReScope(scope) = borrow_data.region {
                        // Check that the scope is not actually a scope from a function that is
                        // a parent of our closure. Note that the CallSite scope itself is
                        // *outside* of the closure, for some weird reason.
                        if let Some(root_scope) = self.root_scope {
                            if *scope != root_scope &&
                                self.scope_tree.is_subscope_of(*scope, root_scope)
                            {
                                sets.kill(&ReserveOrActivateIndex::reserved(borrow_index));
                                if is_activations {
                                    sets.kill(&ReserveOrActivateIndex::active(borrow_index));
                                }
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
            mir::TerminatorKind::Unreachable => {}
        }
    }
}

impl<'a, 'gcx, 'tcx> ActiveBorrows<'a, 'gcx, 'tcx> {
    pub(crate) fn borrows(&self) -> &IndexVec<BorrowIndex, BorrowData<'tcx>> {
        self.0.borrows()
    }

    /// Returns the span for the "end point" given region. This will
    /// return `None` if NLL is enabled, since that concept has no
    /// meaning there.  Otherwise, return region span if it exists and
    /// span for end of the function if it doesn't exist.
    pub(crate) fn opt_region_end_span(&self, region: &Region) -> Option<Span> {
        match self.0.nonlexical_regioncx {
            Some(_) => None,
            None => {
                match self.0.region_span_map.get(region) {
                    Some(span) => Some(span.end_point()),
                    None => Some(self.0.mir.span.end_point())
                }
            }
        }
    }
}

/// `FindPlaceUses` is a MIR visitor that updates `self.sets` for all
/// of the borrows activated by a given statement or terminator.
///
/// ----
///
/// The `ActiveBorrows` flow analysis, when inspecting any given
/// statement or terminator, needs to "generate" (i.e. set to 1) all
/// of the bits for the borrows that are activated by that
/// statement/terminator.
///
/// This struct will seek out all places that are assignment-targets
/// for borrows (gathered in `self.assigned_map`; see also the
/// `assigned_map` in `struct Borrows`), and set the corresponding
/// gen-bits for activations of those borrows in `self.sets`
struct FindPlaceUses<'a, 'b: 'a, 'tcx: 'a> {
    assigned_map: &'a FxHashMap<Place<'tcx>, FxHashSet<BorrowIndex>>,
    sets: &'a mut BlockSets<'b, ReserveOrActivateIndex>,
}

impl<'a, 'b, 'tcx> FindPlaceUses<'a, 'b, 'tcx> {
    fn has_been_reserved(&self, b: &BorrowIndex) -> bool {
        self.sets.on_entry.contains(&ReserveOrActivateIndex::reserved(*b))
    }

    /// return whether `context` should be considered a "use" of a
    /// place found in that context. "Uses" activate associated
    /// borrows (at least when such uses occur while the borrow also
    /// has a reservation at the time).
    fn is_potential_use(context: PlaceContext) -> bool {
        match context {
            // storage effects on an place do not activate it
            PlaceContext::StorageLive | PlaceContext::StorageDead => false,

            // validation effects do not activate an place
            //
            // FIXME: Should they? Is it just another read? Or can we
            // guarantee it won't dereference the stored address? How
            // "deep" does validation go?
            PlaceContext::Validate => false,

            // FIXME: This is here to not change behaviour from before
            // AsmOutput existed, but it's not necessarily a pure overwrite.
            // so it's possible this should activate the place.
            PlaceContext::AsmOutput |
            // pure overwrites of an place do not activate it. (note
            // PlaceContext::Call is solely about dest place)
            PlaceContext::Store | PlaceContext::Call => false,

            // reads of an place *do* activate it
            PlaceContext::Move |
            PlaceContext::Copy |
            PlaceContext::Drop |
            PlaceContext::Inspect |
            PlaceContext::Borrow { .. } |
            PlaceContext::Projection(..) => true,
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for FindPlaceUses<'a, 'b, 'tcx> {
    fn visit_place(&mut self,
                    place: &mir::Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        debug!("FindPlaceUses place: {:?} assigned from borrows: {:?} \
                used in context: {:?} at location: {:?}",
               place, self.assigned_map.get(place), context, location);
        if Self::is_potential_use(context) {
            if let Some(borrows) = self.assigned_map.get(place) {
                for borrow_idx in borrows {
                    debug!("checking if index {:?} for {:?} is reserved ({}) \
                            and thus needs active gen-bit set in sets {:?}",
                           borrow_idx, place, self.has_been_reserved(&borrow_idx), self.sets);
                    if self.has_been_reserved(&borrow_idx) {
                        self.sets.gen(&ReserveOrActivateIndex::active(*borrow_idx));
                    } else {
                        // (This can certainly happen in valid code. I
                        // just want to know about it in the short
                        // term.)
                        debug!("encountered use of Place {:?} of borrow_idx {:?} \
                                at location {:?} outside of reservation",
                               place, borrow_idx, location);
                    }
                }
            }
        }

        self.super_place(place, context, location);
    }
}


impl<'a, 'gcx, 'tcx> BitDenotation for Reservations<'a, 'gcx, 'tcx> {
    type Idx = ReserveOrActivateIndex;
    fn name() -> &'static str { "reservations" }
    fn bits_per_block(&self) -> usize {
        self.0.borrows.len() * 2
    }
    fn start_block_effect(&self, _entry_set: &mut IdxSet<ReserveOrActivateIndex>)  {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect on
        // `_sets`.
    }

    fn before_statement_effect(&self,
                               sets: &mut BlockSets<ReserveOrActivateIndex>,
                               location: Location) {
        debug!("Reservations::before_statement_effect sets: {:?} location: {:?}", sets, location);
        self.0.kill_loans_out_of_scope_at_location(sets, location, false);
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<ReserveOrActivateIndex>,
                        location: Location) {
        debug!("Reservations::statement_effect sets: {:?} location: {:?}", sets, location);
        self.0.statement_effect_on_borrows(sets, location, false);
    }

    fn before_terminator_effect(&self,
                                sets: &mut BlockSets<ReserveOrActivateIndex>,
                                location: Location) {
        debug!("Reservations::before_terminator_effect sets: {:?} location: {:?}", sets, location);
        self.0.kill_loans_out_of_scope_at_location(sets, location, false);
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<ReserveOrActivateIndex>,
                         location: Location) {
        debug!("Reservations::terminator_effect sets: {:?} location: {:?}", sets, location);
        self.0.terminator_effect_on_borrows(sets, location, false);
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

impl<'a, 'gcx, 'tcx> BitDenotation for ActiveBorrows<'a, 'gcx, 'tcx> {
    type Idx = ReserveOrActivateIndex;
    fn name() -> &'static str { "active_borrows" }

    /// Overriding this method; `ActiveBorrows` uses the intrablock
    /// state in `on_entry` to track the current reservations (which
    /// then affect the construction of the gen/kill sets for
    /// activations).
    fn accumulates_intrablock_state() -> bool { true }

    fn bits_per_block(&self) -> usize {
        self.0.borrows.len() * 2
    }

    fn start_block_effect(&self, _entry_sets: &mut IdxSet<ReserveOrActivateIndex>)  {
        // no borrows of code region_scopes have been taken prior to
        // function execution, so this method has no effect on
        // `_sets`.
    }

    fn before_statement_effect(&self,
                               sets: &mut BlockSets<ReserveOrActivateIndex>,
                               location: Location) {
        debug!("ActiveBorrows::before_statement_effect sets: {:?} location: {:?}", sets, location);
        self.0.kill_loans_out_of_scope_at_location(sets, location, true);
    }

    fn statement_effect(&self,
                        sets: &mut BlockSets<ReserveOrActivateIndex>,
                        location: Location) {
        debug!("ActiveBorrows::statement_effect sets: {:?} location: {:?}", sets, location);
        self.0.statement_effect_on_borrows(sets, location, true);
    }

    fn before_terminator_effect(&self,
                                sets: &mut BlockSets<ReserveOrActivateIndex>,
                                location: Location) {
        debug!("ActiveBorrows::before_terminator_effect sets: {:?} location: {:?}", sets, location);
        self.0.kill_loans_out_of_scope_at_location(sets, location, true);
    }

    fn terminator_effect(&self,
                         sets: &mut BlockSets<ReserveOrActivateIndex>,
                         location: Location) {
        debug!("ActiveBorrows::terminator_effect sets: {:?} location: {:?}", sets, location);
        self.0.terminator_effect_on_borrows(sets, location, true);
    }

    fn propagate_call_return(&self,
                             _in_out: &mut IdxSet<ReserveOrActivateIndex>,
                             _call_bb: mir::BasicBlock,
                             _dest_bb: mir::BasicBlock,
                             _dest_place: &mir::Place) {
        // there are no effects on borrows from method call return...
        //
        // ... but If overwriting a place can affect flow state, then
        // latter is not true; see NOTE on Assign case in
        // statement_effect_on_borrows.
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for Reservations<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // union effects of preds when computing reservations
    }
}

impl<'a, 'gcx, 'tcx> BitwiseOperator for ActiveBorrows<'a, 'gcx, 'tcx> {
    #[inline]
    fn join(&self, pred1: usize, pred2: usize) -> usize {
        pred1 | pred2 // union effects of preds when computing activations
    }
}

impl<'a, 'gcx, 'tcx> InitialFlow for Reservations<'a, 'gcx, 'tcx> {
    #[inline]
    fn bottom_value() -> bool {
        false // bottom = no Rvalue::Refs are reserved by default
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
        Static(ref static_) => tcx.is_static_mut(static_.def_id),
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
