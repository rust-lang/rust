//! A pass that simplifies branches when their condition is known.

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

/// The lowering for `if CONST` produces
/// ```
/// _1 = Const(...);
/// switchInt (move _1)
/// ```
/// so this pass replaces that with
/// ```
/// switchInt (Const(...))
/// ```
/// so that further MIR consumers can special-case it more easily.
///
/// Unlike ConstProp, this supports generic constants too, not just concrete ones.
pub struct SimplifyIfConst;

impl<'tcx> MirPass<'tcx> for SimplifyIfConst {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for block in body.basic_blocks_mut() {
            simplify_assign_move_switch(tcx, block);
        }
    }
}

fn simplify_assign_move_switch(tcx: TyCtxt<'_>, block: &mut BasicBlockData<'_>) {
    let Some(Terminator { kind: TerminatorKind::SwitchInt { discr: switch_desc, ..}, ..}) =
        &mut block.terminator
    else { return };

    let &mut Operand::Move(switch_place) = &mut*switch_desc
    else { return };

    let Some(switch_local) = switch_place.as_local()
    else { return };

    let Some(last_statement) = block.statements.last_mut()
    else { return };

    let StatementKind::Assign(boxed_place_rvalue) = &last_statement.kind
    else { return };

    let Some(assigned_local) = boxed_place_rvalue.0.as_local()
    else { return };

    if switch_local != assigned_local {
        return;
    }

    if !matches!(boxed_place_rvalue.1, Rvalue::Use(Operand::Constant(_))) {
        return;
    }

    let should_optimize = tcx.consider_optimizing(|| {
        format!(
            "SimplifyBranches - Assignment: {:?} SourceInfo: {:?}",
            boxed_place_rvalue, last_statement.source_info
        )
    });

    if should_optimize {
        let Some(last_statement) = block.statements.pop()
        else { bug!("Somehow the statement disappeared?"); };

        let StatementKind::Assign(boxed_place_rvalue) = last_statement.kind
        else { bug!("Somehow it's not an assignment any more?"); };

        let Rvalue::Use(assigned_constant @ Operand::Constant(_)) = boxed_place_rvalue.1
        else { bug!("Somehow it's not a use of a constant any more?"); };

        *switch_desc = assigned_constant;
    }
}
