//! A pass that propagates the unreachable terminator of a block to its predecessors
//! when all of their successors are unreachable. This is achieved through a
//! post-order traversal of the blocks.

use rustc_abi::Size;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::bug;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};

use crate::patch::MirPatch;

pub(super) struct UnreachablePropagation;

impl crate::MirPass<'_> for UnreachablePropagation {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // Enable only under -Zmir-opt-level=2 as this can make programs less debuggable.
        sess.mir_opt_level() >= 2
    }

    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut patch = MirPatch::new(body);
        let mut unreachable_blocks = FxHashSet::default();

        for (bb, bb_data) in traversal::postorder(body) {
            let terminator = bb_data.terminator();
            let is_unreachable = match &terminator.kind {
                TerminatorKind::Unreachable => true,
                // This will unconditionally run into an unreachable and is therefore unreachable
                // as well.
                TerminatorKind::Goto { target } if unreachable_blocks.contains(target) => {
                    patch.patch_terminator(bb, TerminatorKind::Unreachable);
                    true
                }
                // Try to remove unreachable targets from the switch.
                TerminatorKind::SwitchInt { .. } => {
                    remove_successors_from_switch(tcx, bb, &unreachable_blocks, body, &mut patch)
                }
                _ => false,
            };
            if is_unreachable {
                unreachable_blocks.insert(bb);
            }
        }

        patch.apply(body);

        // We do want do keep some unreachable blocks, but make them empty.
        // The order in which we clear bb statements does not matter.
        #[allow(rustc::potential_query_instability)]
        for bb in unreachable_blocks {
            body.basic_blocks_mut()[bb].statements.clear();
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

/// Return whether the current terminator is fully unreachable.
fn remove_successors_from_switch<'tcx>(
    tcx: TyCtxt<'tcx>,
    bb: BasicBlock,
    unreachable_blocks: &FxHashSet<BasicBlock>,
    body: &Body<'tcx>,
    patch: &mut MirPatch<'tcx>,
) -> bool {
    let terminator = body.basic_blocks[bb].terminator();
    let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind else { bug!() };
    let source_info = terminator.source_info;
    let location = body.terminator_loc(bb);

    let is_unreachable = |bb| unreachable_blocks.contains(&bb);

    // If there are multiple targets, we want to keep information about reachability for codegen.
    // For example (see tests/codegen-llvm/match-optimizes-away.rs)
    //
    // pub enum Two { A, B }
    // pub fn identity(x: Two) -> Two {
    //     match x {
    //         Two::A => Two::A,
    //         Two::B => Two::B,
    //     }
    // }
    //
    // This generates a `switchInt() -> [0: 0, 1: 1, otherwise: unreachable]`, which allows us or
    // LLVM to turn it into just `x` later. Without the unreachable, such a transformation would be
    // illegal.
    //
    // In order to preserve this information, we record reachable and unreachable targets as
    // `Assume` statements in MIR.

    let discr_ty = discr.ty(body, tcx);
    let discr_size = Size::from_bits(match discr_ty.kind() {
        ty::Uint(uint) => uint.normalize(tcx.sess.target.pointer_width).bit_width().unwrap(),
        ty::Int(int) => int.normalize(tcx.sess.target.pointer_width).bit_width().unwrap(),
        ty::Char => 32,
        ty::Bool => 1,
        other => bug!("unhandled type: {:?}", other),
    });

    let mut add_assumption = |binop, value| {
        let local = patch.new_temp(tcx.types.bool, source_info.span);
        let value = Operand::Constant(Box::new(ConstOperand {
            span: source_info.span,
            user_ty: None,
            const_: Const::from_scalar(tcx, Scalar::from_uint(value, discr_size), discr_ty),
        }));
        let cmp = Rvalue::BinaryOp(binop, Box::new((discr.to_copy(), value)));
        patch.add_assign(location, local.into(), cmp);

        let assume = NonDivergingIntrinsic::Assume(Operand::Move(local.into()));
        patch.add_statement(location, StatementKind::Intrinsic(Box::new(assume)));
    };

    let otherwise = targets.otherwise();
    let otherwise_unreachable = is_unreachable(otherwise);

    let reachable_iter = targets.iter().filter(|&(value, bb)| {
        let is_unreachable = is_unreachable(bb);
        // We remove this target from the switch, so record the inequality using `Assume`.
        if is_unreachable && !otherwise_unreachable {
            add_assumption(BinOp::Ne, value);
        }
        !is_unreachable
    });

    let new_targets = SwitchTargets::new(reachable_iter, otherwise);

    let num_targets = new_targets.all_targets().len();
    let fully_unreachable = num_targets == 1 && otherwise_unreachable;

    let terminator = match (num_targets, otherwise_unreachable) {
        // If all targets are unreachable, we can be unreachable as well.
        (1, true) => TerminatorKind::Unreachable,
        (1, false) => TerminatorKind::Goto { target: otherwise },
        (2, true) => {
            // All targets are unreachable except one. Record the equality, and make it a goto.
            let (value, target) = new_targets.iter().next().unwrap();
            add_assumption(BinOp::Eq, value);
            TerminatorKind::Goto { target }
        }
        _ if num_targets == targets.all_targets().len() => {
            // Nothing has changed.
            return false;
        }
        _ => TerminatorKind::SwitchInt { discr: discr.clone(), targets: new_targets },
    };

    patch.patch_terminator(bb, terminator);
    fully_unreachable
}
