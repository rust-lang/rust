use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{Local, Location, Operand, Statement, StatementKind};
use rustc_middle::ty::TyCtxt;

pub(crate) struct StorageRemover<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) reused_locals: DenseBitSet<Local>,
}

impl<'tcx> MutVisitor<'tcx> for StorageRemover<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_operand(&mut self, operand: &mut Operand<'tcx>, _: Location) {
        if let Operand::Move(place) = *operand
            && !place.is_indirect_first_projection()
            && self.reused_locals.contains(place.local)
        {
            *operand = Operand::Copy(place);
        }
    }

    fn visit_statement(&mut self, stmt: &mut Statement<'tcx>, loc: Location) {
        match stmt.kind {
            // When removing storage statements, we need to remove both (#107511).
            StatementKind::StorageLive(l) | StatementKind::StorageDead(l)
                if self.reused_locals.contains(l) =>
            {
                stmt.make_nop(true)
            }
            _ => self.super_statement(stmt, loc),
        }
    }
}
