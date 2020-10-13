//! A pass that simplifies branches when their condition is known.

use crate::transform::MirPass;
use rustc_data_structures::statistics::Statistic;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use std::borrow::Cow;

pub struct SimplifyBranches {
    label: String,
}

static NUM_TRUE_ASSERT_SIMPLIFIED: Statistic =
    Statistic::new(module_path!(), "Number of true assertions simplified to a goto");
static NUM_FALSE_EDGE_SIMPLIFIED: Statistic =
    Statistic::new(module_path!(), "Number of false edges simplified to a goto");
static NUM_FALSE_UNWIND_SIMPLIFIED: Statistic =
    Statistic::new(module_path!(), "Number of false unwinds simplified to a goto");

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
                } if (c.literal.try_eval_bool(tcx, param_env) == Some(true)) == expected => {
                    NUM_TRUE_ASSERT_SIMPLIFIED.increment(1);
                    TerminatorKind::Goto { target }
                }
                TerminatorKind::FalseEdge { real_target, .. } => {
                    NUM_FALSE_EDGE_SIMPLIFIED.increment(1);
                    TerminatorKind::Goto { target: real_target }
                }
                TerminatorKind::FalseUnwind { real_target, .. } => {
                    NUM_FALSE_UNWIND_SIMPLIFIED.increment(1);
                    TerminatorKind::Goto { target: real_target }
                }
                _ => continue,
            };
        }
    }
}
