use rustc_index::vec::IndexVec;
use rustc_middle::{
    mir::traversal, mir::visit::MutVisitor, mir::visit::PlaceContext, mir::Body, mir::Local,
    mir::Location, mir::Operand, mir::Place, mir::ProjectionElem, mir::Rvalue, mir::Statement,
    mir::StatementKind, ty::TyCtxt,
};
use smallvec::SmallVec;

use super::{MirPass, MirSource};

pub struct Bbcp;

impl<'tcx> MirPass<'tcx> for Bbcp {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        debug!("processing {:?}", source.def_id());

        let mut visitor = BbcpVisitor {
            tcx,
            local_values: IndexVec::from_elem_n(None, body.local_decls.len()),
            invalidation_map: IndexVec::from_elem_n(SmallVec::new(), body.local_decls.len()),
            can_replace: true,
        };

        let reachable = traversal::reachable_as_bitset(body);

        for bb in reachable.iter() {
            visitor.visit_basic_block_data(bb, &mut body.basic_blocks_mut()[bb]);

            for opt in visitor.local_values.iter_mut() {
                *opt = None;
            }

            for inv in visitor.invalidation_map.iter_mut() {
                inv.clear();
            }
        }
    }
}

struct BbcpVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    /// Tracks the (symbolic) values of local variables in the visited block.
    local_values: IndexVec<Local, Option<Place<'tcx>>>,

    /// Maps source locals to a list of destination locals to invalidate when the source is
    /// deallocated or modified.
    invalidation_map: IndexVec<Local, SmallVec<[Local; 4]>>,

    /// Whether we're allowed to apply replacements. This is temporarily set to `false` to avoid
    /// replacing locals whose address is taken.
    can_replace: bool,
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
                if place_eligible(place) {
                    debug!("recording value at {:?}: {:?} = {:?}", location, dest, place);
                    self.local_values[dest] = Some(*place);
                    self.invalidation_map[place.local].push(dest);
                    return;
                }
            }
        }

        self.super_statement(statement, location);
    }

    fn visit_local(&mut self, local: &mut Local, context: PlaceContext, location: Location) {
        // We invalidate a local `l`, and any other locals whose assigned values contain `l`, if:
        // - `l` is mutated (via assignment or other stores, by taking a mutable ref/ptr, etc.)
        // - `l` is reallocated by storage statements (which deinitialized its storage)
        // - `l` is moved from (to avoid use-after-moves)

        if context.is_mutating_use() || context.is_storage_marker() || context.is_move() {
            debug!("invalidation of {:?} at {:?}: {:?} -> clearing", local, location, context);
            self.local_values[*local] = None;
            for inv in &self.invalidation_map[*local] {
                self.local_values[*inv] = None;
            }
            self.invalidation_map[*local].clear();
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, location: Location) {
        // Prevent replacing anything an address is taken from.
        // Doing that would cause code like:
        //     _1 = ...;
        //     _2 = _1;
        //     _3 = &_1;
        //     _4 = &_2;
        // to assign the same address to _3 and _4, which we don't want to do.

        let takes_ref = match rvalue {
            Rvalue::Use(..)
            | Rvalue::Repeat(..)
            | Rvalue::ThreadLocalRef(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::Aggregate(..) => false,
            Rvalue::AddressOf(..) | Rvalue::Ref(..) => true,
        };

        if takes_ref {
            let old_can_replace = self.can_replace;
            self.can_replace = false;
            self.super_rvalue(rvalue, location);
            self.can_replace = old_can_replace;
        } else {
            self.super_rvalue(rvalue, location);
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, location: Location) {
        // We can *not* do:
        //   _1 = ...;
        //   _2 = move _1;
        //   use(move _2);      <- can not replace with `use(_1)`
        // Because `_1` was already moved out of. This is handled when recording values of locals
        // though, so we won't end up here in that situation. When we see a `move` *here*, it must
        // be fine to instead make a *copy* of the original:
        //   _1 = ...;
        //   _2 = _1;
        //   use(move _2);   <- can replace with `use(_2)`

        // NB: All `operand`s are non-mutating uses of the contained place.
        if let Operand::Copy(place) | Operand::Move(place) = operand {
            if self.can_replace {
                if let Some(local) = place.as_local() {
                    if let Some(known_place) = self.local_values[local] {
                        debug!(
                            "{:?}: replacing use of {:?} with {:?}",
                            location, place, known_place
                        );
                        *operand = Operand::Copy(known_place);
                    }
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
