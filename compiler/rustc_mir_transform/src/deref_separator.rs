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
                    let mut place_local = place.local;
                    let mut last_len = 0;
                    for (idx, (p_ref, p_elem)) in place.iter_projections().enumerate() {
                        if p_elem == ProjectionElem::Deref && !p_ref.projection.is_empty() {
                            // The type that we are derefing.
                            let ty = p_ref.ty(local_decl, tcx).ty;
                            let temp = patch.new_temp(ty, stmt.source_info.span);

                            // Because we are assigning this right before original statement
                            // we are using index i of statement.
                            let loc = Location { block: block, statement_index: i };
                            patch.add_statement(loc, StatementKind::StorageLive(temp));

                            // We are adding current p_ref's projections to our
                            // temp value, excluding projections we already covered.
                            let deref_place = Place::from(place_local)
                                .project_deeper(&p_ref.projection[last_len..], tcx);
                            patch.add_assign(
                                loc,
                                Place::from(temp),
                                Rvalue::Use(Operand::Move(deref_place)),
                            );

                            place_local = temp;
                            last_len = p_ref.projection.len();

                            // We are creating a place by using our temp value's location
                            // and copying derefed values which we need to create new statement.
                            let temp_place =
                                Place::from(temp).project_deeper(&place.projection[idx..], tcx);
                            let new_stmt = Statement {
                                source_info: stmt.source_info,
                                kind: StatementKind::Assign(Box::new((
                                    og_place,
                                    Rvalue::Ref(region, borrow_knd, temp_place),
                                ))),
                            };

                            // Replace current statement with newly created one.
                            *stmt = new_stmt;

                            // Since our job with the temp is done it should be gone
                            let loc = Location { block: block, statement_index: i + 1 };
                            patch.add_statement(loc, StatementKind::StorageDead(temp));
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
