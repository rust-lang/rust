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

        'blocks: for (bb, block) in body.basic_blocks.iter_enumerated() {
            let mut pre_local_const: Option<(Local, &'_ ConstOperand<'_>)> = None;

            for (statement_index, stmt) in block.statements.iter().enumerate() {
                let has_local_const = pre_local_const.take();
                // Simplify `assume` of a known value: either a NOP or unreachable.
                if let StatementKind::Intrinsic(box ref intrinsic) = stmt.kind
                    && let NonDivergingIntrinsic::Assume(discr) = intrinsic
                {
                    let c = if let Operand::Constant(c) = discr {
                        c
                    } else if let Some((local, c)) = has_local_const
                        && let Some(assume_local) = discr.place().and_then(|p| p.as_local())
                        && local == assume_local
                    {
                        c
                    } else {
                        continue;
                    };
                    let Some(constant) = c.const_.try_eval_bool(tcx, typing_env) else {
                        continue;
                    };
                    if constant {
                        patch.nop_statement(Location { block: bb, statement_index });
                    } else {
                        patch.patch_terminator(bb, TerminatorKind::Unreachable);
                        continue 'blocks;
                    }
                } else if let StatementKind::Assign(box (ref lhs, ref rvalue)) = stmt.kind
                    && let Some(local) = lhs.as_local()
                    && let Rvalue::Use(Operand::Constant(c)) = rvalue
                    && c.const_.ty().is_bool()
                {
                    pre_local_const = Some((local, c));
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
