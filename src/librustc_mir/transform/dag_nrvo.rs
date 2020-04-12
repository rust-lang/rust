//! NRVO on acyclic CFGs.
//!
//! # Motivation
//!
//! MIR building can insert a lot of redundant copies, and Rust code in general often tends to move
//! values around a lot. The result is a lot of assignments of the form `dest = {move} src;` in MIR.
//! MIR building for constants in particular tends to create additional locals that are only used
//! inside a single block to shuffle a value around unnecessarily.
//!
//! LLVM by itself is not good enough at eliminating these redundant copies (eg. see
//! https://github.com/rust-lang/rust/issues/32966), so this leaves some performance on the table
//! that we can regain by implementing an optimization for removing these assign statements in rustc
//! itself. When this optimization runs fast enough, it can also speed up the constant evaluation
//! and code generation phases of rustc due to the reduced number of statements and locals.
//!
//! # The Optimization
//!
//! Conceptually, this is a Named Return Value Optimization, or NRVO, known from the C++ world. On
//! a very high level, independently of the actual implementation details, it does the following:
//!
//! 1) Identify `dest = src;` statements that can be soundly eliminated.
//! 2) Replace all mentions of `src` with `dest` ("unifying" them).
//! 3) Delete the `dest = src;` statement (by making it a `nop`).
//!
//! Step 1) is by far the hardest, so it is explained in more detail below.
//!
//! ## Soundness
//!
//! Given an `Assign` statement `dest = src;`, where `dest` is a `Place` and `src` is an `Rvalue`,
//! there are a few requirements that must hold for the optimization to be sound:
//!
//! * `dest` must not contain any *indirection* through a pointer. It must access part of the base
//!   local. Otherwise it might point to arbitrary memory that is hard to track.
//!
//!   It must also not contain any indexing projections, since those take an arbitrary `Local` as
//!   the index, and that local might only be initialized shortly before `dest` is used.
//!
//!   Subtle case: If `dest` is a, or projects through a union, then we have to make sure that there
//!   remains an assignment to it, since that sets the "active field" of the union. But if `src` is
//!   a ZST, it might not be initialized, so there might not be any use of it before the assignment,
//!   and performing the optimization would simply delete the assignment, leaving `dest`
//!   uninitialized.
//!
//! * `src` must be a bare `Local` without any indirections or field projections (FIXME: Why).
//!   It can be copied or moved by the assignment.
//!
//! * The `dest` and `src` locals must never be [*live*][liveness] at the same time. If they are, it
//!   means that they both hold a (potentially different) value that is needed by a future use of
//!   the locals. Unifying them would overwrite one of the values.
//!
//!   Note that computing liveness of locals that have had their address taken is more difficult:
//!   Short of doing full escape analysis on the address/pointer/reference, the pass would need to
//!   assume that any operation that can potentially involve opaque user code (such as function
//!   calls, destructors, and inline assembly) may access any local that had its address taken
//!   before that point.
//!
//! Here, the first two conditions are simple structural requirements on the `Assign` statements
//! that can be trivially checked. The liveness requirement however is more difficult and costly to
//! check.
//!
//! ## Approximate Solution
//!
//! A [previous attempt] at implementing an optimization like this turned out to be a significant
//! regression in compiler performance. Fixing the regressions introduced a lot of undesirable
//! complexity to the implementation.
//!
//! For that reason, this pass takes a more conservative approach that might miss some optimization
//! opportunities, but is very simple and runs fast. In particular, it doesn't handle:
//!
//! * **Loops**. It will not optimize functions containing any loops of any kind.
//! * **Borrowing**. If any local has its address taken anywhere inside the function the pass will
//!   not consider it for unification with another local.
//!
//! The implementation works as follows:
//! * Precalculate which locals have their address taken.
//! * Precalculate which blocks use which locals.
//! * Identify candidate assignments of the right form (ensuring all soundness requirements except
//!   liveness).
//! * For each candidate assignment, walk the CFG forwards starting at the `Assign` statement and
//!   collect all uses of `dest`s base local, and walk the CFG *backwards* to collect all uses of
//!   `src`s base local.
//!
//!   These sets are the sets of uses that are [*reachable*][reachability] from the assignment,
//!   respectively the uses that can reach the assignment.
//!
//!   If these sets are equal to the previously calculated *full* sets of uses of
//!   `dest` and `src`, then the live ranges of `src` and `dest` do not overlap and the assignment
//!   can be eliminated.
//!
//! This approximation works only on acyclic CFGs, since all predecessors of the block containing
//! the assignments can only run before the assignment, and all successors must run after it. If
//! there is no use of `dest` before the assignment, and no use of `src` after it, then it follows
//! that the assignment is the point where `src`s live range ends and `dest`s live range starts.
//!
//! ## Pre/Post Optimization
//!
//! This pass is only effective if `SimplifyCfg` runs shortly before it to eliminate unreachable
//! code (as it walks the MIR to find uses of variables).
//!
//! It is recommended to run `SimplifyCfg` and then `SimplifyLocals` some time after this pass, as
//! it replaces the eliminated assign statements with `nop`s and leaves unused locals behind.
//!
//! [liveness]: https://en.wikipedia.org/wiki/Live_variable_analysis
//! [reachability]: https://en.wikipedia.org/wiki/Reachability
//! [previous attempt]: https://github.com/rust-lang/rust/pull/47954
//!
//! Next steps:
//! * Precompute, for every block, the blocks reachable from it (in both forward and backwards
//!   direction), excluding the block itself.
//! * Precompute the list of blocks that use any given local (using a `BitMatrix`).

use crate::transform::{MirPass, MirSource};
use rustc_index::{bit_set::BitSet, vec::IndexVec};
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext, Visitor};
use rustc_middle::mir::{
    read_only, Body, BodyAndCache, Local, LocalKind, Location, Operand, Place, PlaceElem,
    ReadOnlyBodyAndCache, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use smallvec::SmallVec;
use std::collections::VecDeque;

pub struct Nrvo;

impl<'tcx> MirPass<'tcx> for Nrvo {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level == 0 {
            return;
        }

        // We do not handle cyclic CFGs yet.
        if body.is_cfg_cyclic() {
            return;
        }

        if let Some(promoted) = &source.promoted {
            debug!("running on {}-{:?}", tcx.def_path_str(source.def_id()), promoted)
        } else {
            debug!("running on {}", tcx.def_path_str(source.def_id()));
        }

        let mut uses_of = uses_of_locals(body);
        uses_of.iter_mut().for_each(|v| {
            v.sort_unstable();
            v.dedup();
        });

        let mut candidates = find_candidates(tcx, body);
        candidates.retain(|candidate @ &CandidateAssignment { dest, src, loc }| {
            if !tcx.consider_optimizing(|| format!("DAG-NRVO {:?}", candidate)) {
                return false;
            }

            // In order to merge `dest` with `src`, this assignment must be:
            // - The first thing writing to `dest`.
            // - The last thing reading from `src`.
            // Where `first` and `last` have to consider all possible control flows inside the
            // function. Since we only handle acyclic CFGs for now, this is as simple as traversing
            // the CFG: Starting at the assignment location, we must be able to reach all other
            // uses of `src` by moving backwards through the CFG, and all other uses of `dest` by
            // moving forwards.

            let past_src_uses =
                uses_relative_to_location(src, loc, Direction::Backward, &read_only!(body));
            let future_dest_uses =
                uses_relative_to_location(dest.local, loc, Direction::Forward, &read_only!(body));

            debug!("{:?} = {:?} at {:?}", dest, src, loc);
            debug!("dest: uses_of[{:?}] = {:?}", dest.local, uses_of[dest.local]);
            debug!("future_dest_uses  = {:?}", future_dest_uses);
            debug!("src:  uses_of[{:?}] = {:?}", src, uses_of[src]);
            debug!("past_src_uses     = {:?}", past_src_uses);

            if uses_of[dest.local] != future_dest_uses || uses_of[src] != past_src_uses {
                debug!("(ineligible)");
                return false;
            }

            true
        });

        // At this point they stop being just "candidates", really.

        let mut replacements = IndexVec::from_elem_n(None, body.local_decls.len());
        for CandidateAssignment { dest, src, loc } in candidates {
            assert!(replacements[src].is_none(), "multiple applicable candidates for {:?}", src);
            replacements[src] = Some(dest);

            body.make_statement_nop(loc);
        }

        // First, we might have replacements that themselves contain to-be-replaced locals.
        // Substitute them first so that the replacer doesn't have to do it redundantly.
        let mut dest_locals = BitSet::new_empty(body.local_decls.len());
        for local in replacements.indices() {
            if let Some(replacement) = replacements[local] {
                // Collect all locals in `dest` to later delete their storage statements.
                dest_locals.insert(replacement.local);

                // Substitute the base local of `replacement` until fixpoint.
                let mut base = replacement.local;
                let mut reversed_projection_slices = Vec::with_capacity(1);
                while let Some(replacement_for_replacement) = replacements[base] {
                    base = replacement_for_replacement.local;
                    reversed_projection_slices.push(replacement_for_replacement.projection);
                }

                let projection: Vec<_> = reversed_projection_slices
                    .iter()
                    .rev()
                    .flat_map(|projs| projs.iter().copied())
                    .chain(replacement.projection.iter().copied())
                    .collect();
                let projection = tcx.intern_place_elems(&projection);

                // FIXME check what happens if `src` is contained in `dest` (infinite loop?)
                // (we need to reject that case, but it may also happen across multiple pairs)

                // Replace with the final `Place`.
                replacements[local] = Some(Place { local: base, projection });
            }
        }

        debug!("replacements {:?}", replacements);

        // Replace the `src` locals with the `dest` place, and erase storage statements for `src`
        // and `dest`s base local.
        Replacer { tcx, map: replacements, kill: dest_locals, place_elem_cache: Vec::new() }
            .visit_body(body);

        // FIXME fix debug info
    }
}

struct Replacer<'tcx> {
    tcx: TyCtxt<'tcx>,
    map: IndexVec<Local, Option<Place<'tcx>>>,
    kill: BitSet<Local>,
    place_elem_cache: Vec<PlaceElem<'tcx>>,
}

impl<'tcx> MutVisitor<'tcx> for Replacer<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_local(&mut self, local: &mut Local, context: PlaceContext, location: Location) {
        if context.is_use() && self.map[*local].is_some() {
            bug!(
                "use of local {:?} should have been replaced by visit_place; context={:?}, loc={:?}",
                local,
                context,
                location,
            );
        }
    }

    fn process_projection_elem(
        &mut self,
        elem: &PlaceElem<'tcx>,
        _: Location,
    ) -> Option<PlaceElem<'tcx>> {
        match elem {
            PlaceElem::Index(local) => {
                if let Some(replacement) = self.map[*local] {
                    bug!(
                        "cannot replace {:?} with {:?} in index projection {:?}",
                        local,
                        replacement,
                        elem,
                    );
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let Some(replacement) = self.map[place.local] {
            // Rebase `place`s projections onto `self.with`.
            self.place_elem_cache.clear(); // FIXME faster
            self.place_elem_cache.extend(replacement.projection);
            self.place_elem_cache.extend(place.projection);
            let projection = self.tcx.intern_place_elems(&self.place_elem_cache);
            let new_place = Place { local: replacement.local, projection };

            debug!("Replacer: {:?} -> {:?}", place, new_place);
            *place = new_place;
        }

        self.super_place(place, context, location);
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match &statement.kind {
            // FIXME: Don't delete storage statements, merge the live ranges instead
            StatementKind::StorageDead(local) | StatementKind::StorageLive(local)
                if self.map[*local].is_some() || self.kill.contains(*local) =>
            {
                statement.make_nop()
            }
            _ => self.super_statement(statement, location),
        }
    }
}

enum Direction {
    Forward,
    Backward,
}

/// Starting at the `Assign` statement at `loc` and moving in `dir`, collects all reachable uses of
/// `local`.
fn uses_relative_to_location(
    local: Local,
    loc: Location,
    dir: Direction,
    body: &ReadOnlyBodyAndCache<'_, '_>,
) -> SmallVec<[Location; 1]> {
    let mut uses = SmallVec::new();
    // The assignment at `loc` always contributes a use, regardless of `dir`.
    uses.push(loc);

    let mut collector = UseCollector {
        callback: |current_local, location| {
            if current_local == local {
                uses.push(location);
            }
        },
    };
    let mut seen = BitSet::new_empty(body.basic_blocks().len());

    seen.insert(loc.block);

    let statements = &body.basic_blocks()[loc.block].statements;

    // FIXME: Must visit the missing half of the assign stmt too!
    // (actually this might not work since `Location` isn't granular enough if there's a use on
    // both sides of the assignment)

    // Process statements before/after `loc` in the starting block.
    let stmt_range = match dir {
        Direction::Forward => loc.statement_index + 1..statements.len(),
        Direction::Backward => 0..loc.statement_index,
    };
    for (statement, index) in statements[stmt_range.clone()].iter().zip(stmt_range) {
        collector.visit_statement(statement, Location { block: loc.block, statement_index: index });
    }

    if let Direction::Forward = dir {
        collector.visit_terminator(
            body.basic_blocks()[loc.block].terminator(),
            body.terminator_loc(loc.block),
        );
    }

    let mut work_queue = VecDeque::new();
    let enqueue_next_blocks = |block, queue: &mut VecDeque<_>| match dir {
        Direction::Forward => {
            queue.extend(body.basic_blocks()[block].terminator().successors().copied());
        }
        Direction::Backward => {
            queue.extend(body.predecessors_for(block));
        }
    };

    enqueue_next_blocks(loc.block, &mut work_queue);

    while let Some(block) = work_queue.pop_front() {
        if !seen.insert(block) {
            continue; // already seen
        }

        collector.visit_basic_block_data(block, &body.basic_blocks()[block]);
        enqueue_next_blocks(block, &mut work_queue);
    }

    uses.sort_unstable();
    uses.dedup();
    uses
}

/// A visitor that invokes a callback when any local is used in a way that's relevant to this
/// analysis.
struct UseCollector<F> {
    callback: F,
}

impl<'tcx, F> Visitor<'tcx> for UseCollector<F>
where
    F: FnMut(Local, Location),
{
    fn visit_local(&mut self, local: &Local, context: PlaceContext, location: Location) {
        // This gets called on debuginfo, so check that the context is actually a use.
        if context.is_use() {
            (self.callback)(*local, location);
        }
    }

    fn visit_projection_elem(
        &mut self,
        _local: Local,
        _proj_base: &[PlaceElem<'tcx>],
        elem: &PlaceElem<'tcx>,
        _context: PlaceContext,
        location: Location,
    ) {
        if let PlaceElem::Index(local) = elem {
            (self.callback)(*local, location);
        }
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        // Do not visit storage statements as those will be removed/inferred after merging.
        match &statement.kind {
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
            _ => self.super_statement(statement, location),
        }
    }
}

/// Collects all locations where any local is accessed in any way.
fn uses_of_locals(body: &Body<'_>) -> IndexVec<Local, SmallVec<[Location; 1]>> {
    let mut uses = IndexVec::from_elem_n(SmallVec::new(), body.local_decls.len());
    let mut collector = UseCollector {
        callback: |local, location| {
            uses[local].push(location);
        },
    };
    collector.visit_body(body);
    uses
}

/// A `dest = {move} src;` statement at `loc`.
///
/// We want to consider merging `dest` and `src` due to this assignment.
#[derive(Debug)]
struct CandidateAssignment<'tcx> {
    dest: Place<'tcx>,
    src: Local,
    loc: Location,
}

fn find_candidates<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
) -> Vec<CandidateAssignment<'tcx>> {
    struct FindAssignments<'a, 'tcx> {
        tcx: TyCtxt<'tcx>,
        body: &'a Body<'tcx>,
        candidates: Vec<CandidateAssignment<'tcx>>,
        ever_borrowed_locals: BitSet<Local>,
        locals_used_as_array_index: BitSet<Local>,
    }

    impl<'a, 'tcx> Visitor<'tcx> for FindAssignments<'a, 'tcx> {
        fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
            if let StatementKind::Assign(box (
                dest,
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src)),
            )) = &statement.kind
            {
                // `dest` must not have pointer indirection.
                if dest.is_indirect() {
                    return;
                }

                // `src` must be a plain local.
                if !src.projection.is_empty() {
                    return;
                }

                // Since we want to replace `src` with `dest`, `src` must not be required.
                if is_local_required(src.local, self.body) {
                    return;
                }

                // Neither `src` nor `dest` must ever have its address taken.
                if self.ever_borrowed_locals.contains(dest.local)
                    || self.ever_borrowed_locals.contains(src.local)
                {
                    return;
                }

                assert_ne!(dest.local, src.local, "self-assignments are UB");

                // We can't replace locals occurring in `PlaceElem::Index` for now.
                if self.locals_used_as_array_index.contains(src.local) {
                    return;
                }

                // Handle the "subtle case" described above by rejecting any `dest` that is or
                // projects through a union.
                let is_union = |ty: Ty<'_>| {
                    if let ty::Adt(def, _) = &ty.kind {
                        if def.is_union() {
                            return true;
                        }
                    }

                    false
                };
                let mut place_ty = PlaceTy::from_ty(self.body.local_decls[dest.local].ty);
                if is_union(place_ty.ty) {
                    return;
                }
                for elem in dest.projection {
                    if let PlaceElem::Index(_) = elem {
                        // `dest` contains an indexing projection.
                        return;
                    }

                    place_ty = place_ty.projection_ty(self.tcx, elem);
                    if is_union(place_ty.ty) {
                        return;
                    }
                }

                self.candidates.push(CandidateAssignment {
                    dest: *dest,
                    src: src.local,
                    loc: location,
                });
            }
        }
    }

    let mut visitor = FindAssignments {
        tcx,
        body,
        candidates: Vec::new(),
        ever_borrowed_locals: ever_borrowed_locals(body),
        locals_used_as_array_index: locals_used_as_array_index(body),
    };
    visitor.visit_body(body);
    visitor.candidates
}

/// Some locals are part of the function's interface and can not be removed.
///
/// Note that these locals *can* still be merged with non-required locals by removing that other
/// local.
fn is_local_required(local: Local, body: &Body<'_>) -> bool {
    match body.local_kind(local) {
        LocalKind::Arg | LocalKind::ReturnPointer => true,
        LocalKind::Var | LocalKind::Temp => false,
    }
}

/// Walks MIR to find all locals that have their address taken anywhere.
fn ever_borrowed_locals(body: &Body<'_>) -> BitSet<Local> {
    struct BorrowCollector {
        locals: BitSet<Local>,
    }

    impl<'tcx> Visitor<'tcx> for BorrowCollector {
        fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
            self.super_rvalue(rvalue, location);

            match rvalue {
                Rvalue::AddressOf(_, borrowed_place) | Rvalue::Ref(_, _, borrowed_place) => {
                    if !borrowed_place.is_indirect() {
                        self.locals.insert(borrowed_place.local);
                    }
                }

                Rvalue::Cast(..)
                | Rvalue::Use(..)
                | Rvalue::Repeat(..)
                | Rvalue::Len(..)
                | Rvalue::BinaryOp(..)
                | Rvalue::CheckedBinaryOp(..)
                | Rvalue::NullaryOp(..)
                | Rvalue::UnaryOp(..)
                | Rvalue::Discriminant(..)
                | Rvalue::Aggregate(..) => {}
            }
        }

        fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
            self.super_terminator(terminator, location);

            match terminator.kind {
                TerminatorKind::Drop { location: dropped_place, .. }
                | TerminatorKind::DropAndReplace { location: dropped_place, .. } => {
                    self.locals.insert(dropped_place.local);
                }

                TerminatorKind::Abort
                | TerminatorKind::Assert { .. }
                | TerminatorKind::Call { .. }
                | TerminatorKind::FalseEdges { .. }
                | TerminatorKind::FalseUnwind { .. }
                | TerminatorKind::GeneratorDrop
                | TerminatorKind::Goto { .. }
                | TerminatorKind::Resume
                | TerminatorKind::Return
                | TerminatorKind::SwitchInt { .. }
                | TerminatorKind::Unreachable
                | TerminatorKind::Yield { .. } => {}
            }
        }
    }

    let mut visitor = BorrowCollector { locals: BitSet::new_empty(body.local_decls.len()) };
    visitor.visit_body(body);
    visitor.locals
}

/// `PlaceElem::Index` only stores a `Local`, so we can't replace that with a full `Place`.
///
/// Collect locals used as indices so we don't generate candidates that are impossible to apply
/// later.
///
/// FIXME `PlaceElem::Index` should probably removed completely, they're a pain
fn locals_used_as_array_index(body: &Body<'_>) -> BitSet<Local> {
    struct IndexCollector {
        locals: BitSet<Local>,
    }

    impl<'tcx> Visitor<'tcx> for IndexCollector {
        fn visit_projection_elem(
            &mut self,
            local: Local,
            proj_base: &[PlaceElem<'tcx>],
            elem: &PlaceElem<'tcx>,
            context: PlaceContext,
            location: Location,
        ) {
            if let PlaceElem::Index(i) = *elem {
                self.locals.insert(i);
            }
            self.super_projection_elem(local, proj_base, elem, context, location);
        }
    }

    let mut visitor = IndexCollector { locals: BitSet::new_empty(body.local_decls.len()) };
    visitor.visit_body(body);
    visitor.locals
}
