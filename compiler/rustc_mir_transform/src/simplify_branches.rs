use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub enum SimplifyConstCondition {
    AfterConstProp,
    Final,
}
/// A pass that replaces a branch with a goto when its condition is known.
impl<'tcx> MirPass<'tcx> for SimplifyConstCondition {
    fn name(&self) -> &'static str {
        match self {
            SimplifyConstCondition::AfterConstProp => "SimplifyConstCondition-after-const-prop",
            SimplifyConstCondition::Final => "SimplifyConstCondition-final",
        }
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running SimplifyConstCondition on {:?}", body.source);
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        let ssa = crate::ssa::SsaLocals::new(body);
        'blocks: for block in body.basic_blocks_mut() {
            for stmt in block.statements.iter_mut() {
                if let StatementKind::Intrinsic(box ref intrinsic) = stmt.kind
                    && let NonDivergingIntrinsic::Assume(discr) = intrinsic
                    && let Operand::Constant(ref c) = discr
                    && let Some(constant) = c.const_.try_eval_bool(tcx, param_env)
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

            block.terminator_mut().kind = match block.terminator().kind {
                TerminatorKind::SwitchInt { ref discr, ref targets, .. } => match discr {
                    Operand::Move(place) if let Some(local) = place.as_local() => {
                        if !ssa.is_ssa(local) {
                            continue;
                        }
                        let Some(c) = block.statements.iter().find_map(|stmt| {
                            if let StatementKind::Assign(box (
                                assigned,
                                Rvalue::Use(Operand::Constant(const_val)),
                            )) = &stmt.kind
                                && assigned == place
                            {
                                Some(const_val)
                            } else {
                                None
                            }
                        }) else {
                            continue;
                        };
                        let constant = c.const_.try_eval_bits(tcx, param_env);
                        if let Some(constant) = constant {
                            let target = targets.target_for_value(constant);
                            TerminatorKind::Goto { target }
                        } else {
                            continue;
                        }
                    }
                    Operand::Constant(ref c) => {
                        let constant = c.const_.try_eval_bits(tcx, param_env);
                        if let Some(constant) = constant {
                            let target = targets.target_for_value(constant);
                            TerminatorKind::Goto { target }
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                },
                TerminatorKind::Assert {
                    target, cond: Operand::Constant(ref c), expected, ..
                } => match c.const_.try_eval_bool(tcx, param_env) {
                    Some(v) if v == expected => TerminatorKind::Goto { target },
                    _ => continue,
                },
                _ => continue,
            };
        }
    }
}
