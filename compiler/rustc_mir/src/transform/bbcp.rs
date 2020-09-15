use rustc_index::{bit_set::BitSet, vec::IndexVec};
use rustc_middle::{
    mir::traversal, mir::visit::MutVisitor, mir::visit::PlaceContext, mir::visit::Visitor,
    mir::Body, mir::Local, mir::Location, mir::Operand, mir::Place, mir::ProjectionElem,
    mir::Rvalue, mir::Statement, mir::StatementKind, mir::Terminator, mir::TerminatorKind,
    ty::TyCtxt,
};
use smallvec::SmallVec;

use super::{MirPass, MirSource};

pub struct Bbcp;

impl<'tcx> MirPass<'tcx> for Bbcp {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        debug!("processing {:?}", source.def_id());

        let mut borrows = BorrowCollector { locals: BitSet::new_empty(body.local_decls.len()) };
        borrows.visit_body(body);

        let mut visitor = BbcpVisitor {
            tcx,
            referenced_locals: borrows.locals,
            local_values: LocalValues::new(body),
        };

        let reachable = traversal::reachable_as_bitset(body);

        for bb in reachable.iter() {
            visitor.visit_basic_block_data(bb, &mut body.basic_blocks_mut()[bb]);
            visitor.local_values.clear();
        }
    }
}

/// Symbolic value of a local variable.
#[derive(Copy, Clone)]
enum LocalValue<'tcx> {
    Unknown,

    /// The local is definitely assigned to `place`.
    Place {
        place: Place<'tcx>,
        generation: u32,
    },
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

struct BbcpVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Locals that have their address taken. We will not replace those, since that may change their
    /// observed address.
    referenced_locals: BitSet<Local>,

    /// Tracks the (symbolic) values of local variables in the visited block.
    local_values: LocalValues<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for BbcpVisitor<'tcx> {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
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
        // We can *not* do:
        //   _1 = ...;
        //   _2 = move _1;
        //   use(move _2);   <- can not replace with `use(move _1)` or `use(_1)`
        // Because `_1` was already moved out of. This is handled when recording values of locals
        // though, so we won't end up here in that situation. When we see a `move` *here*, it must
        // be fine to instead make a *copy* of the original:
        //   _1 = ...;
        //   _2 = _1;
        //   use(move _2);   <- can replace with `use(_1)`

        // NB: All `operand`s are non-mutating uses of the contained place.
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

fn place_eligible(place: &Place<'_>) -> bool {
    place.projection.iter().all(|elem| match elem {
        ProjectionElem::Deref | ProjectionElem::Index(_) => false,

        ProjectionElem::Field(..)
        | ProjectionElem::ConstantIndex { .. }
        | ProjectionElem::Subslice { .. }
        | ProjectionElem::Downcast(..) => true,
    })
}

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
