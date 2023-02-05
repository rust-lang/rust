use crate::dataflow_const_prop::DataflowConstProp;
use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

/// A pass that replaces a branch with a goto when its condition is known.
pub struct SimplifyConstCondition {
    label: String,
}

impl SimplifyConstCondition {
    pub fn new(label: &str) -> Self {
        SimplifyConstCondition { label: format!("SimplifyConstCondition-{}", label) }
    }
}

impl<'tcx> MirPass<'tcx> for SimplifyConstCondition {
    fn name(&self) -> &str {
        &self.label
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        loop {
            let mut changed = false;
            let param_env = tcx.param_env(body.source.def_id());
            for block in body.basic_blocks_mut() {
                let terminator = block.terminator_mut();
                terminator.kind = match terminator.kind {
                    TerminatorKind::SwitchInt {
                        discr: Operand::Constant(ref c),
                        ref targets,
                        ..
                    } => {
                        let constant = c.literal.try_eval_bits(tcx, param_env, c.ty());
                        if let Some(constant) = constant {
                            changed = true;
                            let target = targets.target_for_value(constant);
                            TerminatorKind::Goto { target }
                        } else {
                            continue;
                        }
                    }
                    TerminatorKind::Assert {
                        target,
                        cond: Operand::Constant(ref c),
                        expected,
                        ..
                    } => match c.literal.try_eval_bool(tcx, param_env) {
                        Some(v) if v == expected => {
                            changed = true;
                            TerminatorKind::Goto { target }
                        }
                        _ => continue,
                    },
                    _ => continue,
                };
            }
            if !changed { break } else { DataflowConstProp.run_pass(tcx, body) }
        }
    }
}
