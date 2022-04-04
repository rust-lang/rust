use crate::MirPass;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
pub struct Derefer;

pub fn deref_finder<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let mut patch = MirPatch::new(body);
    let (basic_blocks, local_decl) = body.basic_blocks_and_local_decls_mut();
    for (block, data) in basic_blocks.iter_enumerated_mut() {
        for (i, stmt) in data.statements.iter_mut().enumerate() {
            match stmt.kind {
                StatementKind::Assign(box (og_place, Rvalue::Ref(region, borrow_knd, place))) => {
                    if borrow_knd == (BorrowKind::Mut { allow_two_phase_borrow: false }) {
                        for (idx, (p_ref, p_elem)) in place.iter_projections().enumerate() {
                            if p_elem == ProjectionElem::Deref {
                                // The type that we are derefing
                                let ty = p_ref.ty(local_decl, tcx).ty;
                                let temp = patch.new_temp(ty, stmt.source_info.span);

                                // Because we are assigning this right before original statement
                                // we are using index i of statement
                                let loc = Location { block: block, statement_index: i };
                                patch.add_statement(loc, StatementKind::StorageLive(temp));

                                // We are adding current p_ref's projections to our
                                // temp value
                                let deref_place =
                                    Place::from(p_ref.local).project_deeper(p_ref.projection, tcx);
                                patch.add_assign(
                                    loc,
                                    Place::from(temp),
                                    Rvalue::Use(Operand::Move(deref_place)),
                                );

                                // We are creating a place by using our temp value's location
                                // and copying derefed values we need to it
                                let temp_place =
                                    Place::from(temp).project_deeper(&place.projection[idx..], tcx);
                                patch.add_assign(
                                    loc,
                                    og_place,
                                    Rvalue::Ref(region, borrow_knd, temp_place),
                                );
                                // We have to delete the original statement since we just
                                // replaced it
                                stmt.make_nop();
                            }
                        }
                    }
                }
                _ => (),
            }
        }
    }
    patch.apply(body);
}

impl<'tcx> MirPass<'tcx> for Derefer {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        deref_finder(tcx, body);
    }
}
