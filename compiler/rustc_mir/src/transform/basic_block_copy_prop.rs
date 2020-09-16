//! A simple intra-block copy propagation pass.
//!
//! This pass performs simple forwards-propagation of locals that were assigned within the same MIR
//! block. This is a common pattern introduced by MIR building.
//!
//! The pass is fairly simple: It walks every MIR block from top to bottom and looks for assignments
//! we can track. At the same time, it looks for uses of locals whose value we have previously
//! recorded, and replaces them with their value.
//!
//! "Value" in this case means `LocalValue`, not an actual concrete value (that would be constant
//! propagation). `LocalValue` is either `Unknown`, which means that a local has no value we can
//! substitute it for, or `Place`, which means that the local was previously assigned to a copy of
//! some `Place` and can be replaced by it.
//!
//! Removal of the left-over assignments and locals (if possible) is performed by the
//! `SimplifyLocals` pass that runs later.
//!
//! The pass has one interesting optimization to ensure that it runs in linear time: Recorded values
//! are tagged by a "generation", which indicates in which basic block the value was recorded.
//! `LocalValues` will only return values that were recorded in the current generation (so in the
//! current block). Once we move on to a different block, we bump the generation counter, which will
//! result in all old values becoming inaccessible. This is logically equivalent to simply
//! overwriting all of them with `Unknown`, but is much cheaper: Just an increment instead of an
//! `O(n)` clear (where `n` is the number of locals declared in the body). This `O(n)` runtime would
//! otherwise make the total runtime of this pass `O(n * m)`, where `n` is the number of locals and
//! `m` is the number of basic blocks, which is prohibitively expensive.

use rustc_index::{bit_set::BitSet, vec::IndexVec};
use rustc_middle::{
    mir::visit::MutVisitor, mir::visit::PlaceContext, mir::visit::Visitor, mir::Body, mir::Local,
    mir::Location, mir::Operand, mir::Place, mir::ProjectionElem, mir::Rvalue, mir::Statement,
    mir::StatementKind, mir::Terminator, mir::TerminatorKind, ty::TyCtxt,
};
use smallvec::SmallVec;

use super::{MirPass, MirSource};

pub struct BasicBlockCopyProp;

impl<'tcx> MirPass<'tcx> for BasicBlockCopyProp {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level == 0 {
            return;
        }

        debug!("processing {:?}", source.def_id());

        let mut borrows = BorrowCollector { locals: BitSet::new_empty(body.local_decls.len()) };
        borrows.visit_body(body);

        let mut visitor = CopyPropVisitor {
            tcx,
            referenced_locals: borrows.locals,
            local_values: LocalValues::new(body),
        };

        for (bb, data) in body.basic_blocks_mut().iter_enumerated_mut() {
            visitor.visit_basic_block_data(bb, data);
            visitor.local_values.clear();
        }
    }
}

/// Symbolic value of a local variable.
#[derive(Copy, Clone)]
enum LocalValue<'tcx> {
    /// Locals start out with unknown values, and are assigned an unknown value when they are
    /// mutated in an incompatible way.
    Unknown,

    /// The local was last assigned to a copy of `place`.
    ///
    /// If a local is in this state, and we see a use of that local, we can substitute `place`
    /// instead, potentially eliminating the local and its assignment.
    Place { place: Place<'tcx>, generation: u32 },
}

/// Stores the locals that need to be invalidated when this local is modified or deallocated.
#[derive(Default, Clone)]
struct Invalidation {
    locals: SmallVec<[Local; 4]>,
    generation: u32,
}

struct LocalValues<'tcx> {
    /// Tracks the values that were assigned to local variables in the current basic block.
    map: IndexVec<Local, LocalValue<'tcx>>,

    /// Maps source locals to a list of destination locals to invalidate when the source is
    /// deallocated or modified.
    invalidation_map: IndexVec<Local, Invalidation>,

    /// Data generation in this map.
    ///
    /// This is bumped when entering a new block. When looking data up, we ensure it was stored in
    /// the same generation. This allows clearing the map by simply incrementing the generation
    /// instead of having to clear the data (which can perform poorly).
    generation: u32,
}

impl<'tcx> LocalValues<'tcx> {
    fn new(body: &Body<'_>) -> Self {
        Self {
            map: IndexVec::from_elem_n(LocalValue::Unknown, body.local_decls.len()),
            invalidation_map: IndexVec::from_elem_n(
                Invalidation::default(),
                body.local_decls.len(),
            ),
            generation: 0,
        }
    }

    fn get(&self, local: Local) -> Option<Place<'tcx>> {
        match self.map[local] {
            LocalValue::Place { place, generation } if generation == self.generation => Some(place),
            _ => None,
        }
    }

    /// Records an assignment of `place` to `local`.
    fn insert(&mut self, local: Local, place: Place<'tcx>) {
        self.map[local] = LocalValue::Place { place, generation: self.generation };

        let inval = &mut self.invalidation_map[place.local];
        if inval.generation != self.generation {
            inval.locals.clear();
            inval.generation = self.generation;
        }

        inval.locals.push(local);
    }

    /// Resets `local`'s state to `Unknown` and invalidates all locals that have been assigned to
    /// `local` in the past.
    fn invalidate(&mut self, local: Local) {
        self.map[local] = LocalValue::Unknown;

        let inval = &mut self.invalidation_map[local];
        if inval.generation == self.generation {
            for local in &inval.locals {
                self.map[*local] = LocalValue::Unknown;
            }

            inval.locals.clear();
        }
    }

    /// Marks the data in the map as dirty, effectively clearing it.
    fn clear(&mut self) {
        self.generation += 1;
    }
}

struct CopyPropVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Locals that have their address taken.
    referenced_locals: BitSet<Local>,

    /// Tracks the (symbolic) values of local variables in the visited block.
    local_values: LocalValues<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for CopyPropVisitor<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        // We are only looking for *copies* of a place, not moves, since moving out of the place
        // *again* later on would be a use-after-move. For example:
        //   _1 = ...;
        //   _2 = move _1;
        //   use(move _2);   <- can *not* replace with `use(move _1)` or `use(_1)`
        if let StatementKind::Assign(box (dest, Rvalue::Use(Operand::Copy(place)))) =
            &statement.kind
        {
            if let Some(dest) = dest.as_local() {
                if place_eligible(place)
                    && !self.referenced_locals.contains(dest)
                    && !self.referenced_locals.contains(place.local)
                {
                    debug!("recording value at {:?}: {:?} = {:?}", location, dest, place);
                    self.local_values.insert(dest, *place);
                    return;
                }
            }
        }

        // If this is not an eligible assignment to be recorded, visit it. This will keep track of
        // any mutations of locals via `visit_local` and assign an `Unknown` value to them.
        self.super_statement(statement, location);
    }

    fn visit_local(&mut self, local: &mut Local, context: PlaceContext, location: Location) {
        // We invalidate a local `l`, and any other locals whose assigned values contain `l`, if:
        // - `l` is mutated (via assignment or other stores, by taking a mutable ref/ptr, etc.)
        // - `l` is reallocated by storage statements (which deinitializes its storage)
        // - `l` is moved from (to avoid use-after-moves)

        if context.is_mutating_use() || context.is_storage_marker() || context.is_move() {
            debug!("invalidation of {:?} at {:?}: {:?} -> clearing", local, location, context);
            self.local_values.invalidate(*local);
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        // NB: All `operand`s are non-mutating uses of the contained place, so we don't have to call
        // `super_operand` here.

        // We *can* replace a `move` by a copy from the recorded place, because we only record
        // places that are *copied* from in the first place (so the place type must be `Copy` by
        // virtue of the input MIR). For example, this is a common pattern:
        //   _1 = ...;
        //   _2 = _1;
        //   use(move _2);   <- can replace with `use(_1)`
        if let Operand::Copy(place) | Operand::Move(place) = operand {
            if let Some(local) = place.as_local() {
                if let Some(known_place) = self.local_values.get(local) {
                    debug!("{:?}: replacing use of {:?} with {:?}", location, place, known_place);
                    *operand = Operand::Copy(known_place);
                }
            }
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

/// Collects locals that have their address taken.
///
/// We do not optimize such locals since they can be modified through operations that do not mention
/// the local. Doing so might also change the local's address, which is observable.
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
            | Rvalue::Aggregate(..)
            | Rvalue::ThreadLocalRef(..) => {}
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.super_terminator(terminator, location);

        match terminator.kind {
            TerminatorKind::Drop { place: dropped_place, .. }
            | TerminatorKind::DropAndReplace { place: dropped_place, .. } => {
                self.locals.insert(dropped_place.local);
            }

            TerminatorKind::Abort
            | TerminatorKind::Assert { .. }
            | TerminatorKind::Call { .. }
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Goto { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Return
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Unreachable
            | TerminatorKind::Yield { .. }
            | TerminatorKind::InlineAsm { .. } => {}
        }
    }
}
