use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::trace;

use crate::patch::MirPatch;

pub(super) enum SimplifyConstCondition {
    AfterConstProp,
    Final,
}

/// A pass that replaces a branch with a goto when its condition is known.
impl<'tcx> crate::MirPass<'tcx> for SimplifyConstCondition {
    fn name(&self) -> &'static str {
        match self {
            SimplifyConstCondition::AfterConstProp => "SimplifyConstCondition-after-const-prop",
            SimplifyConstCondition::Final => "SimplifyConstCondition-final",
        }
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running SimplifyConstCondition on {:?}", body.source);
        let typing_env = body.typing_env(tcx);
        let mut patch = MirPatch::new(body);

        'blocks: for (bb, block) in body.basic_blocks.iter_enumerated() {
            for (statement_index, stmt) in block.statements.iter().enumerate() {
                // Simplify `assume` of a known value: either a NOP or unreachable.
                if let StatementKind::Intrinsic(box ref intrinsic) = stmt.kind
                    && let NonDivergingIntrinsic::Assume(discr) = intrinsic
                    && let Operand::Constant(c) = discr
                    && let Some(constant) = c.const_.try_eval_bool(tcx, typing_env)
                {
                    if constant {
                        patch.nop_statement(Location { block: bb, statement_index });
                    } else {
                        patch.patch_terminator(bb, TerminatorKind::Unreachable);
                        continue 'blocks;
                    }
                }
            }

            let terminator = block.terminator();
            let terminator = match terminator.kind {
                TerminatorKind::SwitchInt {
                    discr: Operand::Constant(ref c), ref targets, ..
                } => {
                    let constant = c.const_.try_eval_bits(tcx, typing_env);
                    if let Some(constant) = constant {
                        let target = targets.target_for_value(constant);
                        TerminatorKind::Goto { target }
                    } else {
                        continue;
                    }
                }
                TerminatorKind::Assert {
                    target, cond: Operand::Constant(ref c), expected, ..
                } => match c.const_.try_eval_bool(tcx, typing_env) {
                    Some(v) if v == expected => TerminatorKind::Goto { target },
                    _ => continue,
                },
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
