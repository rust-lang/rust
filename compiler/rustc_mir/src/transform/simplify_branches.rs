//! A pass that simplifies branches when their condition is known.

use crate::transform::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use std::borrow::Cow;

pub struct SimplifyBranches {
    label: String,
}

impl SimplifyBranches {
    pub fn new(label: &str) -> Self {
        SimplifyBranches { label: format!("SimplifyBranches-{}", label) }
    }
}

impl<'tcx> MirPass<'tcx> for SimplifyBranches {
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
                        let otherwise = targets.otherwise();
                        let mut ret = TerminatorKind::Goto { target: otherwise };
                        for (v, t) in targets.iter() {
                            if v == constant {
                                ret = TerminatorKind::Goto { target: t };
                                break;
                            }
                        }
                        ret
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
                TerminatorKind::FalseEdge { real_target, .. } => {
                    TerminatorKind::Goto { target: real_target }
                }
                TerminatorKind::FalseUnwind { real_target, .. } => {
                    TerminatorKind::Goto { target: real_target }
                }
                _ => continue,
            };
        }
    }
}
