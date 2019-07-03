//! A pass that simplifies branches when their condition is known.

use rustc::ty::{TyCtxt, ParamEnv};
use rustc::mir::*;
use crate::transform::{MirPass, MirSource};

use std::borrow::Cow;

pub struct SimplifyBranches { label: String }

impl SimplifyBranches {
    pub fn new(label: &str) -> Self {
        SimplifyBranches { label: format!("SimplifyBranches-{}", label) }
    }
}

impl MirPass for SimplifyBranches {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.label)
    }

    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        for block in body.basic_blocks_mut() {
            let terminator = block.terminator_mut();
            terminator.kind = match terminator.kind {
                TerminatorKind::SwitchInt {
                    discr: Operand::Constant(ref c), switch_ty, ref values, ref targets, ..
                } => {
                    let switch_ty = ParamEnv::empty().and(switch_ty);
                    let constant = c.literal.assert_bits(tcx, switch_ty);
                    if let Some(constant) = constant {
                        let (otherwise, targets) = targets.split_last().unwrap();
                        let mut ret = TerminatorKind::Goto { target: *otherwise };
                        for (&v, t) in values.iter().zip(targets.iter()) {
                            if v == constant {
                                ret = TerminatorKind::Goto { target: *t };
                                break;
                            }
                        }
                        ret
                    } else {
                        continue
                    }
                },
                TerminatorKind::Assert {
                    target, cond: Operand::Constant(ref c), expected, ..
                } if (c.literal.assert_bool(tcx) == Some(true)) == expected =>
                    TerminatorKind::Goto { target },
                TerminatorKind::FalseEdges { real_target, .. } => {
                    TerminatorKind::Goto { target: real_target }
                },
                TerminatorKind::FalseUnwind { real_target, .. } => {
                    TerminatorKind::Goto { target: real_target }
                },
                _ => continue
            };
        }
    }
}
