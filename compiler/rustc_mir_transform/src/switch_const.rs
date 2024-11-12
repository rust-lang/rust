//! A pass that makes `SwitchInt`-on-`const` more obvious to later code.

use rustc_middle::bug;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

/// A `MirPass` for simplifying `if T::CONST`.
///
/// Today, MIR building for things like `if T::IS_ZST` introduce a constant
/// for the copy of the bool, so it ends up in MIR as
/// `_1 = CONST; switchInt (move _1)` or `_2 = CONST; switchInt (_2)`.
///
/// This pass is very specifically targeted at *exactly* those patterns.
/// It can absolutely be replaced with a more general pass should we get one that
/// we can run in low optimization levels, but at the time of writing even in
/// optimized builds this wasn't simplified.
#[derive(Default)]
pub struct SwitchConst;

impl<'tcx> MirPass<'tcx> for SwitchConst {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        for block in body.basic_blocks.as_mut_preserves_cfg() {
            let switch_local = if let TerminatorKind::SwitchInt { discr, .. } =
                &block.terminator().kind
                && let Some(place) = discr.place()
                && let Some(local) = place.as_local()
            {
                local
            } else {
                continue;
            };

            let new_operand = if let Some(statement) = block.statements.last()
                && let StatementKind::Assign(place_and_rvalue) = &statement.kind
                && let Some(local) = place_and_rvalue.0.as_local()
                && local == switch_local
                && let Rvalue::Use(operand) = &place_and_rvalue.1
                && let Operand::Constant(_) = operand
            {
                operand.clone()
            } else {
                continue;
            };

            if !tcx.consider_optimizing(|| format!("SwitchConst: switchInt(move {switch_local:?}"))
            {
                break;
            }

            let TerminatorKind::SwitchInt { discr, .. } = &mut block.terminator_mut().kind else {
                bug!("Somehow wasn't a switchInt any more?")
            };
            *discr = new_operand;
        }
    }
}
