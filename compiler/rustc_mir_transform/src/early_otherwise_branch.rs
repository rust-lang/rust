use std::fmt::Debug;

use rustc_middle::mir::*;
use rustc_middle::ty::{Ty, TyCtxt};
use tracing::trace;

use super::simplify::simplify_cfg;
use crate::patch::MirPatch;

/// This pass optimizes something like
/// ```ignore (syntax-highlighting-only)
/// let x: Option<()>;
/// let y: Option<()>;
/// match (x,y) {
///     (Some(_), Some(_)) => {0},
///     (None, None) => {2},
///     _ => {1}
/// }
/// ```
/// into something like
/// ```ignore (syntax-highlighting-only)
/// let x: Option<()>;
/// let y: Option<()>;
/// let discriminant_x = std::mem::discriminant(x);
/// let discriminant_y = std::mem::discriminant(y);
/// if discriminant_x == discriminant_y {
///     match x {
///         Some(_) => 0,
///         None => 2,
///     }
/// } else {
///     1
/// }
/// ```
///
/// Specifically, it looks for instances of control flow like this:
/// ```text
///
///     =================
///     |      BB1      |
///     |---------------|                  ============================
///     |     ...       |         /------> |            BBC           |
///     |---------------|         |        |--------------------------|
///     |  switchInt(Q) |         |        |   _cl = discriminant(P)  |
///     |       c       | --------/        |--------------------------|
///     |       d       | -------\         |       switchInt(_cl)     |
///     |      ...      |        |         |            c             | ---> BBC.2
///     |    otherwise  | --\    |    /--- |         otherwise        |
///     =================   |    |    |    ============================
///                         |    |    |
///     =================   |    |    |
///     |      BBU      | <-|    |    |    ============================
///     |---------------|        \-------> |            BBD           |
///     |---------------|             |    |--------------------------|
///     |  unreachable  |             |    |   _dl = discriminant(P)  |
///     =================             |    |--------------------------|
///                                   |    |       switchInt(_dl)     |
///     =================             |    |            d             | ---> BBD.2
///     |      BB9      | <--------------- |         otherwise        |
///     |---------------|                  ============================
///     |      ...      |
///     =================
/// ```
/// Where the `otherwise` branch on `BB1` is permitted to either go to `BBU`. In the
/// code:
///  - `BB1` is `parent` and `BBC, BBD` are children
///  - `P` is `child_place`
///  - `child_ty` is the type of `_cl`.
///  - `Q` is `parent_op`.
///  - `parent_ty` is the type of `Q`.
///  - `BB9` is `destination`
/// All this is then transformed into:
/// ```text
///
///     =======================
///     |          BB1        |
///     |---------------------|                  ============================
///     |          ...        |         /------> |           BBEq           |
///     | _s = discriminant(P)|         |        |--------------------------|
///     | _t = Ne(Q, _s)      |         |        |--------------------------|
///     |---------------------|         |        |       switchInt(Q)       |
///     |     switchInt(_t)   |         |        |            c             | ---> BBC.2
///     |        false        | --------/        |            d             | ---> BBD.2
///     |       otherwise     |       /--------- |         otherwise        |
///     =======================       |          ============================
///                                   |
///     =================             |
///     |      BB9      | <-----------/
///     |---------------|
///     |      ...      |
///     =================
/// ```
pub(super) struct EarlyOtherwiseBranch;

impl<'tcx> crate::MirPass<'tcx> for EarlyOtherwiseBranch {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("running EarlyOtherwiseBranch on {:?}", body.source);

        let mut should_cleanup = false;

        // Also consider newly generated bbs in the same pass
        for parent in body.basic_blocks.indices() {
            let bbs = &*body.basic_blocks;
            let Some(opt_data) = evaluate_candidate(tcx, body, parent) else { continue };

            trace!("SUCCESS: found optimization possibility to apply: {opt_data:?}");

            should_cleanup = true;

            let TerminatorKind::SwitchInt { discr: parent_op, targets: parent_targets } =
                &bbs[parent].terminator().kind
            else {
                unreachable!()
            };
            // Always correct since we can only switch on `Copy` types
            let parent_op = match parent_op {
                Operand::Move(x) => Operand::Copy(*x),
                Operand::Copy(x) => Operand::Copy(*x),
                Operand::Constant(x) => Operand::Constant(x.clone()),
            };
            let parent_ty = parent_op.ty(body.local_decls(), tcx);
            let statements_before = bbs[parent].statements.len();
            let parent_end = Location { block: parent, statement_index: statements_before };

            let mut patch = MirPatch::new(body);

            let (second_discriminant_temp, second_operand) = if opt_data.need_hoist_discriminant {
                // create temp to store second discriminant in, `_s` in example above
                let second_discriminant_temp =
                    patch.new_temp(opt_data.child_ty, opt_data.child_source.span);

                patch.add_statement(
                    parent_end,
                    StatementKind::StorageLive(second_discriminant_temp),
                );

                // create assignment of discriminant
                patch.add_assign(
                    parent_end,
                    Place::from(second_discriminant_temp),
                    Rvalue::Discriminant(opt_data.child_place),
                );
                (
                    Some(second_discriminant_temp),
                    Operand::Move(Place::from(second_discriminant_temp)),
                )
            } else {
                (None, Operand::Copy(opt_data.child_place))
            };

            // create temp to store inequality comparison between the two discriminants, `_t` in
            // example above
            let nequal = BinOp::Ne;
            let comp_res_type = nequal.ty(tcx, parent_ty, opt_data.child_ty);
            let comp_temp = patch.new_temp(comp_res_type, opt_data.child_source.span);
            patch.add_statement(parent_end, StatementKind::StorageLive(comp_temp));

            // create inequality comparison
            let comp_rvalue =
                Rvalue::BinaryOp(nequal, Box::new((parent_op.clone(), second_operand)));
            patch.add_statement(
                parent_end,
                StatementKind::Assign(Box::new((Place::from(comp_temp), comp_rvalue))),
            );

            let eq_new_targets = parent_targets.iter().map(|(value, child)| {
                let TerminatorKind::SwitchInt { targets, .. } = &bbs[child].terminator().kind
                else {
                    unreachable!()
                };
                (value, targets.target_for_value(value))
            });
            // The otherwise either is the same target branch or an unreachable.
            let eq_targets = SwitchTargets::new(eq_new_targets, parent_targets.otherwise());

            // Create `bbEq` in example above
            let eq_switch = BasicBlockData::new(
                Some(Terminator {
                    source_info: bbs[parent].terminator().source_info,
                    kind: TerminatorKind::SwitchInt {
                        // switch on the first discriminant, so we can mark the second one as dead
                        discr: parent_op,
                        targets: eq_targets,
                    },
                }),
                bbs[parent].is_cleanup,
            );

            let eq_bb = patch.new_block(eq_switch);

            // Jump to it on the basis of the inequality comparison
            let true_case = opt_data.destination;
            let false_case = eq_bb;
            patch.patch_terminator(
                parent,
                TerminatorKind::if_(Operand::Move(Place::from(comp_temp)), true_case, false_case),
            );

            if let Some(second_discriminant_temp) = second_discriminant_temp {
                // generate StorageDead for the second_discriminant_temp not in use anymore
                patch.add_statement(
                    parent_end,
                    StatementKind::StorageDead(second_discriminant_temp),
                );
            }

            // Generate a StorageDead for comp_temp in each of the targets, since we moved it into
            // the switch
            for bb in [false_case, true_case].iter() {
                patch.add_statement(
                    Location { block: *bb, statement_index: 0 },
                    StatementKind::StorageDead(comp_temp),
                );
            }

            patch.apply(body);
        }

        // Since this optimization adds new basic blocks and invalidates others,
        // clean up the cfg to make it nicer for other passes
        if should_cleanup {
            simplify_cfg(tcx, body);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

#[derive(Debug)]
struct OptimizationData<'tcx> {
    destination: BasicBlock,
    child_place: Place<'tcx>,
    child_ty: Ty<'tcx>,
    child_source: SourceInfo,
    need_hoist_discriminant: bool,
}

fn evaluate_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    parent: BasicBlock,
) -> Option<OptimizationData<'tcx>> {
    let bbs = &body.basic_blocks;
    // NB: If this BB is a cleanup, we may need to figure out what else needs to be handled.
    if bbs[parent].is_cleanup {
        return None;
    }
    let TerminatorKind::SwitchInt { targets, discr: parent_discr } = &bbs[parent].terminator().kind
    else {
        return None;
    };
    let parent_ty = parent_discr.ty(body.local_decls(), tcx);
    let (_, child) = targets.iter().next()?;

    let Terminator {
        kind: TerminatorKind::SwitchInt { targets: child_targets, discr: child_discr },
        source_info,
    } = bbs[child].terminator()
    else {
        return None;
    };
    let child_ty = child_discr.ty(body.local_decls(), tcx);
    if child_ty != parent_ty {
        return None;
    }

    // We only handle:
    // ```
    // bb4: {
    //     _8 = discriminant((_3.1: Enum1));
    //    switchInt(move _8) -> [2: bb7, otherwise: bb1];
    // }
    // ```
    // and
    // ```
    // bb2: {
    //     switchInt((_3.1: u64)) -> [1: bb5, otherwise: bb1];
    // }
    // ```
    if bbs[child].statements.len() > 1 {
        return None;
    }

    // When thie BB has exactly one statement, this statement should be discriminant.
    let need_hoist_discriminant = bbs[child].statements.len() == 1;
    let child_place = if need_hoist_discriminant {
        if !bbs[targets.otherwise()].is_empty_unreachable() {
            // Someone could write code like this:
            // ```rust
            // let Q = val;
            // if discriminant(P) == otherwise {
            //     let ptr = &mut Q as *mut _ as *mut u8;
            //     // It may be difficult for us to effectively determine whether values are valid.
            //     // Invalid values can come from all sorts of corners.
            //     unsafe { *ptr = 10; }
            // }
            //
            // match P {
            //    A => match Q {
            //        A => {
            //            // code
            //        }
            //        _ => {
            //            // don't use Q
            //        }
            //    }
            //    _ => {
            //        // don't use Q
            //    }
            // };
            // ```
            //
            // Hoisting the `discriminant(Q)` out of the `A` arm causes us to compute the discriminant of an
            // invalid value, which is UB.
            // In order to fix this, **we would either need to show that the discriminant computation of
            // `place` is computed in all branches**.
            // FIXME(#95162) For the moment, we adopt a conservative approach and
            // consider only the `otherwise` branch has no statements and an unreachable terminator.
            return None;
        }
        // Handle:
        // ```
        // bb4: {
        //     _8 = discriminant((_3.1: Enum1));
        //    switchInt(move _8) -> [2: bb7, otherwise: bb1];
        // }
        // ```
        let [
            Statement {
                kind: StatementKind::Assign(box (_, Rvalue::Discriminant(child_place))),
                ..
            },
        ] = bbs[child].statements.as_slice()
        else {
            return None;
        };
        *child_place
    } else {
        // Handle:
        // ```
        // bb2: {
        //     switchInt((_3.1: u64)) -> [1: bb5, otherwise: bb1];
        // }
        // ```
        let Operand::Copy(child_place) = child_discr else {
            return None;
        };
        *child_place
    };
    let destination = if need_hoist_discriminant || bbs[targets.otherwise()].is_empty_unreachable()
    {
        child_targets.otherwise()
    } else {
        targets.otherwise()
    };

    // Verify that the optimization is legal for each branch
    for (value, child) in targets.iter() {
        if !verify_candidate_branch(
            &bbs[child],
            value,
            child_place,
            destination,
            need_hoist_discriminant,
        ) {
            return None;
        }
    }
    Some(OptimizationData {
        destination,
        child_place,
        child_ty,
        child_source: *source_info,
        need_hoist_discriminant,
    })
}

fn verify_candidate_branch<'tcx>(
    branch: &BasicBlockData<'tcx>,
    value: u128,
    place: Place<'tcx>,
    destination: BasicBlock,
    need_hoist_discriminant: bool,
) -> bool {
    // In order for the optimization to be correct, the terminator must be a `SwitchInt`.
    let TerminatorKind::SwitchInt { discr: switch_op, targets } = &branch.terminator().kind else {
        return false;
    };
    if need_hoist_discriminant {
        // If we need hoist discriminant, the branch must have exactly one statement.
        let [statement] = branch.statements.as_slice() else {
            return false;
        };
        // The statement must assign the discriminant of `place`.
        let StatementKind::Assign(box (discr_place, Rvalue::Discriminant(from_place))) =
            statement.kind
        else {
            return false;
        };
        if from_place != place {
            return false;
        }
        // The assignment must invalidate a local that terminate on a `SwitchInt`.
        if !discr_place.projection.is_empty() || *switch_op != Operand::Move(discr_place) {
            return false;
        }
    } else {
        // If we don't need hoist discriminant, the branch must not have any statements.
        if !branch.statements.is_empty() {
            return false;
        }
        // The place on `SwitchInt` must be the same.
        if *switch_op != Operand::Copy(place) {
            return false;
        }
    }
    // It must fall through to `destination` if the switch misses.
    if destination != targets.otherwise() {
        return false;
    }
    // It must have exactly one branch for value `value` and have no more branches.
    let mut iter = targets.iter();
    let (Some((target_value, _)), None) = (iter.next(), iter.next()) else {
        return false;
    };
    target_value == value
}
