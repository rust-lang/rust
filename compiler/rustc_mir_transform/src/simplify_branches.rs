use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use std::borrow::Cow;

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
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.label)
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env(body.source.def_id());
        for block in body.basic_blocks_mut() {
            let terminator = block.terminator_mut();
            terminator.kind = match terminator.kind {
                TerminatorKind::SwitchInt {
                    discr: Operand::Constant(ref c),
                    switch_ty,
                    ref targets,
                    ..
                } => {
                    let constant = c.literal.try_eval_bits(tcx, param_env, switch_ty);
                    if let Some(constant) = constant {
                        let target = targets.target_for_value(constant);
                        TerminatorKind::Goto { target }
                    } else {
                        continue;
                    }
                }
                TerminatorKind::Assert {
                    target, cond: Operand::Constant(ref c), expected, ..
                } => match c.literal.try_eval_bool(tcx, param_env) {
                    Some(v) if v == expected => TerminatorKind::Goto { target },
                    _ => continue,
                },
                _ => continue,
            };
        }
    }
}
