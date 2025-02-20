use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::trace;

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
        'blocks: for block in body.basic_blocks_mut() {
            for stmt in block.statements.iter_mut() {
                // Simplify `assume` of a known value: either a NOP or unreachable.
                if let StatementKind::Intrinsic(box ref intrinsic) = stmt.kind
                    && let NonDivergingIntrinsic::Assume(discr) = intrinsic
                    && let Operand::Constant(c) = discr
                    && let Some(constant) = c.const_.try_eval_bool(tcx, typing_env)
                {
                    if constant {
                        stmt.make_nop();
                    } else {
                        block.statements.clear();
                        block.terminator_mut().kind = TerminatorKind::Unreachable;
                        continue 'blocks;
                    }
                }
            }

            let terminator = block.terminator_mut();
            terminator.kind = match terminator.kind {
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
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}
