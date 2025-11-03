use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::trace;

use crate::patch::MirPatch;

pub(super) enum SimplifyConstCondition {
    AfterInstSimplify,
    AfterConstProp,
    Final,
}

/// A pass that replaces a branch with a goto when its condition is known.
impl<'tcx> crate::MirPass<'tcx> for SimplifyConstCondition {
    fn name(&self) -> &'static str {
        match self {
            SimplifyConstCondition::AfterInstSimplify => {
                "SimplifyConstCondition-after-inst-simplify"
            }
            SimplifyConstCondition::AfterConstProp => "SimplifyConstCondition-after-const-prop",
            SimplifyConstCondition::Final => "SimplifyConstCondition-final",
        }
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running SimplifyConstCondition on {:?}", body.source);
        let typing_env = body.typing_env(tcx);
        let mut patch = MirPatch::new(body);

        fn try_get_const<'tcx, 'a>(
            operand: &'a Operand<'tcx>,
            has_place_const: Option<(Place<'tcx>, &'a ConstOperand<'tcx>)>,
        ) -> Option<&'a ConstOperand<'tcx>> {
            match operand {
                Operand::Constant(const_operand) => Some(const_operand),
                // `has_place_const` must be the LHS of the previous statement.
                // Soundness: There is nothing can modify the place, as there are no statements between the two statements.
                Operand::Copy(place) | Operand::Move(place)
                    if let Some((place_const, const_operand)) = has_place_const
                        && place_const == *place =>
                {
                    Some(const_operand)
                }
                Operand::Copy(_) | Operand::Move(_) => None,
            }
        }

        'blocks: for (bb, block) in body.basic_blocks.iter_enumerated() {
            let mut pre_place_const: Option<(Place<'tcx>, &ConstOperand<'tcx>)> = None;

            for (statement_index, stmt) in block.statements.iter().enumerate() {
                let has_place_const = pre_place_const.take();
                // Simplify `assume` of a known value: either a NOP or unreachable.
                if let StatementKind::Intrinsic(box ref intrinsic) = stmt.kind
                    && let NonDivergingIntrinsic::Assume(discr) = intrinsic
                    && let Some(c) = try_get_const(discr, has_place_const)
                    && let Some(constant) = c.const_.try_eval_bool(tcx, typing_env)
                {
                    if constant {
                        patch.nop_statement(Location { block: bb, statement_index });
                    } else {
                        patch.patch_terminator(bb, TerminatorKind::Unreachable);
                        continue 'blocks;
                    }
                } else if let StatementKind::Assign(box (lhs, ref rvalue)) = stmt.kind
                    && let Rvalue::Use(Operand::Constant(c)) = rvalue
                {
                    pre_place_const = Some((lhs, c));
                }
            }

            let terminator = block.terminator();
            let terminator = match terminator.kind {
                TerminatorKind::SwitchInt { ref discr, ref targets, .. }
                    if let Some(c) = try_get_const(discr, pre_place_const.take())
                        && let Some(constant) = c.const_.try_eval_bits(tcx, typing_env) =>
                {
                    let target = targets.target_for_value(constant);
                    TerminatorKind::Goto { target }
                }
                TerminatorKind::Assert { target, ref cond, expected, .. }
                    if let Some(c) = try_get_const(&cond, pre_place_const.take())
                        && let Some(constant) = c.const_.try_eval_bool(tcx, typing_env)
                        && constant == expected =>
                {
                    TerminatorKind::Goto { target }
                }
                _ => continue,
            };
            patch.patch_terminator(bb, terminator);
        }
        patch.apply(body);
    }

    fn is_required(&self) -> bool {
        false
    }
}
