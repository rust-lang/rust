use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;

/// Walks MIR to find all locals that have their address taken anywhere.
pub fn ever_borrowed_locals(body: &Body<'_>) -> BitSet<Local> {
    let mut visitor = BorrowCollector { locals: BitSet::new_empty(body.local_decls.len()) };
    for (block, data) in body.basic_blocks().iter_enumerated() {
        visitor.visit_basic_block_data(block, data);
    }
    visitor.locals
}

struct BorrowCollector {
    locals: BitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for BorrowCollector {
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, _location: Location) {
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
