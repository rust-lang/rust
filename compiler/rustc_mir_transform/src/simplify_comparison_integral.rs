use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::{
    BinOp, Body, Location, Operand, Place, Rvalue, StatementKind, SwitchTargets, TerminatorKind,
};
use rustc_middle::ty::{Ty, TyCtxt};
use tracing::trace;

use crate::patch::MirPatch;
use crate::ssa::SsaLocals;
use crate::storage_remover::StorageRemover;

/// Pass to convert `if` conditions on integrals into switches on the integral.
/// For an example, it turns something like
///
/// ```ignore (MIR)
/// _3 = Eq(move _4, const 43i32);
/// StorageDead(_4);
/// switchInt(_3) -> [false: bb2, otherwise: bb3];
/// ```
///
/// into:
///
/// ```ignore (MIR)
/// switchInt(_4) -> [43i32: bb3, otherwise: bb2];
/// ```
pub(super) struct SimplifyComparisonIntegral;

impl<'tcx> crate::MirPass<'tcx> for SimplifyComparisonIntegral {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running SimplifyComparisonIntegral on {:?}", body.source);

        let typing_env = body.typing_env(tcx);
        let ssa = SsaLocals::new(tcx, body, typing_env);

        let mut reused_locals = DenseBitSet::new_empty(body.local_decls.len());
        let mut patch = MirPatch::new(body);
        let mut changed = false;
        for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
            let Some((is_move, targets, location)) =
                candidate_switch_int(body, &ssa, &bb_data.terminator().kind)
            else {
                continue;
            };
            let StatementKind::Assign(box (
                _,
                Rvalue::BinaryOp(op @ (BinOp::Eq | BinOp::Ne), box (left, right)),
            )) = &body.basic_blocks[location.block].statements[location.statement_index].kind
            else {
                continue;
            };
            let Some((branch_value_scalar, branch_value_ty, to_switch_on)) =
                find_branch_value_info(left, right, &ssa)
            else {
                continue;
            };
            let new_value = match branch_value_scalar {
                Scalar::Int(int) => {
                    let layout = tcx
                        .layout_of(typing_env.as_query_input(branch_value_ty))
                        .expect("if we have an evaluated constant we must know the layout");
                    int.to_bits(layout.size)
                }
                Scalar::Ptr(..) => continue,
            };
            const FALSE: u128 = 0;
            let mut new_targets = targets.clone();
            let first_value = new_targets.iter().next().unwrap().0;
            let first_is_false_target = first_value == FALSE;
            match (op, first_is_false_target) {
                (BinOp::Eq, true) | (BinOp::Ne, false) => {
                    // If the assignment was Eq we want the true case to be first,
                    // Or If the assignment was Ne we want the false case to be first.
                    new_targets.all_targets_mut().swap(0, 1);
                }
                _ => {}
            }
            new_targets.all_values_mut()[0] = new_value.into();

            if is_move {
                patch.nop_statement(location);
            }
            reused_locals.insert(to_switch_on.local);
            patch.patch_terminator(
                bb,
                TerminatorKind::SwitchInt {
                    discr: Operand::Copy(to_switch_on),
                    targets: new_targets,
                },
            );
            changed = true;
        }
        if changed {
            patch.apply(body);
            StorageRemover { tcx, reused_locals }.visit_body_preserves_cfg(body);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn find_branch_value_info<'tcx>(
    left: &Operand<'tcx>,
    right: &Operand<'tcx>,
    ssa: &SsaLocals,
) -> Option<(Scalar, Ty<'tcx>, Place<'tcx>)> {
    // check that either left or right is a constant.
    // if any are, we can use the other to switch on, and the constant as a value in a switch
    use Operand::*;
    match (left, right) {
        (Constant(branch_value), Copy(to_switch_on) | Move(to_switch_on))
        | (Copy(to_switch_on) | Move(to_switch_on), Constant(branch_value)) => {
            // Make sure that the place is not modified.
            if !ssa.is_ssa(to_switch_on.local) || !to_switch_on.is_stable_offset() {
                return None;
            }
            let branch_value_ty = branch_value.const_.ty();
            // we only want to apply this optimization if we are matching on integrals (and chars),
            // as it is not possible to switch on floats
            if !branch_value_ty.is_integral() && !branch_value_ty.is_char() {
                return None;
            };
            let branch_value_scalar = branch_value.const_.try_to_scalar()?;
            Some((branch_value_scalar, branch_value_ty, *to_switch_on))
        }
        _ => None,
    }
}

fn candidate_switch_int<'tcx, 'body>(
    body: &'body Body<'tcx>,
    ssa: &SsaLocals,
    terminator: &'body TerminatorKind<'tcx>,
) -> Option<(bool, &'body SwitchTargets, Location)> {
    let (discr, targets) = terminator.as_switch()?;
    let local_on_switch = discr.place()?.as_local()?;
    if !body.local_decls[local_on_switch].ty.is_bool() {
        return None;
    }
    let assign_location = ssa.ssa_assign_location(local_on_switch)?;
    Some((discr.is_move(), targets, assign_location))
}
