use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use std::fmt::Debug;

use super::simplify::simplify_cfg;

/// This pass optimizes something like
/// ```ignore (syntax-highlighting-only)
/// let x: Option<()>;
/// let y: Option<()>;
/// match (x,y) {
///     (Some(_), Some(_)) => {0},
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
///         _ => 1, // <----
///     } //               | Actually the same bb
/// } else { //            |
///     1 // <--------------
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
///     |---------------|   |    \-------> |            BBD           |
///     |---------------|   |         |    |--------------------------|
///     |  unreachable  |   |         |    |   _dl = discriminant(P)  |
///     =================   |         |    |--------------------------|
///                         |         |    |       switchInt(_dl)     |
///     =================   |         |    |            d             | ---> BBD.2
///     |      BB9      | <--------------- |         otherwise        |
///     |---------------|                  ============================
///     |      ...      |
///     =================
/// ```
/// Where the `otherwise` branch on `BB1` is permitted to either go to `BBU` or to `BB9`. In the
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
///     |       otherwise     | ---------------- |         otherwise        |
///     =======================       |          ============================
///                                   |
///     =================             |
///     |      BB9      | <-----------/
///     |---------------|
///     |      ...      |
///     =================
/// ```
///
/// This is only correct for some `P`, since `P` is now computed outside the original `switchInt`.
/// The filter on which `P` are allowed (together with discussion of its correctness) is found in
/// `may_hoist`.
pub struct EarlyOtherwiseBranch;

impl<'tcx> MirPass<'tcx> for EarlyOtherwiseBranch {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 3 && sess.opts.unstable_opts.unsound_mir_opts
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("running EarlyOtherwiseBranch on {:?}", body.source);

        let mut should_cleanup = false;

        // Also consider newly generated bbs in the same pass
        for i in 0..body.basic_blocks.len() {
            let bbs = &*body.basic_blocks;
            let parent = BasicBlock::from_usize(i);
            let Some(opt_data) = evaluate_candidate(tcx, body, parent) else {
                continue
            };

            if !tcx.consider_optimizing(|| format!("EarlyOtherwiseBranch {:?}", &opt_data)) {
                break;
            }

            trace!("SUCCESS: found optimization possibility to apply: {:?}", &opt_data);

            should_cleanup = true;

            let TerminatorKind::SwitchInt {
                discr: parent_op,
                switch_ty: parent_ty,
                targets: parent_targets
            } = &bbs[parent].terminator().kind else {
                unreachable!()
            };
            // Always correct since we can only switch on `Copy` types
            let parent_op = match parent_op {
                Operand::Move(x) => Operand::Copy(*x),
                Operand::Copy(x) => Operand::Copy(*x),
                Operand::Constant(x) => Operand::Constant(x.clone()),
            };
            let statements_before = bbs[parent].statements.len();
            let parent_end = Location { block: parent, statement_index: statements_before };

            let mut patch = MirPatch::new(body);

            // create temp to store second discriminant in, `_s` in example above
            let second_discriminant_temp =
                patch.new_temp(opt_data.child_ty, opt_data.child_source.span);

            patch.add_statement(parent_end, StatementKind::StorageLive(second_discriminant_temp));

            // create assignment of discriminant
            patch.add_assign(
                parent_end,
                Place::from(second_discriminant_temp),
                Rvalue::Discriminant(opt_data.child_place),
            );

            // create temp to store inequality comparison between the two discriminants, `_t` in
            // example above
            let nequal = BinOp::Ne;
            let comp_res_type = nequal.ty(tcx, *parent_ty, opt_data.child_ty);
            let comp_temp = patch.new_temp(comp_res_type, opt_data.child_source.span);
            patch.add_statement(parent_end, StatementKind::StorageLive(comp_temp));

            // create inequality comparison between the two discriminants
            let comp_rvalue = Rvalue::BinaryOp(
                nequal,
                Box::new((parent_op.clone(), Operand::Move(Place::from(second_discriminant_temp)))),
            );
            patch.add_statement(
                parent_end,
                StatementKind::Assign(Box::new((Place::from(comp_temp), comp_rvalue))),
            );

            let eq_new_targets = parent_targets.iter().map(|(value, child)| {
                let TerminatorKind::SwitchInt{ targets, .. } = &bbs[child].terminator().kind else {
                    unreachable!()
                };
                (value, targets.target_for_value(value))
            });
            let eq_targets = SwitchTargets::new(eq_new_targets, opt_data.destination);

            // Create `bbEq` in example above
            let eq_switch = BasicBlockData::new(Some(Terminator {
                source_info: bbs[parent].terminator().source_info,
                kind: TerminatorKind::SwitchInt {
                    // switch on the first discriminant, so we can mark the second one as dead
                    discr: parent_op,
                    switch_ty: opt_data.child_ty,
                    targets: eq_targets,
                },
            }));

            let eq_bb = patch.new_block(eq_switch);

            // Jump to it on the basis of the inequality comparison
            let true_case = opt_data.destination;
            let false_case = eq_bb;
            patch.patch_terminator(
                parent,
                TerminatorKind::if_(
                    tcx,
                    Operand::Move(Place::from(comp_temp)),
                    true_case,
                    false_case,
                ),
            );

            // generate StorageDead for the second_discriminant_temp not in use anymore
            patch.add_statement(parent_end, StatementKind::StorageDead(second_discriminant_temp));

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
}

/// Returns true if computing the discriminant of `place` may be hoisted out of the branch
fn may_hoist<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, place: Place<'tcx>) -> bool {
    // FIXME(JakobDegen): This is unsound. Someone could write code like this:
    // ```rust
    // let Q = val;
    // if discriminant(P) == otherwise {
    //     let ptr = &mut Q as *mut _ as *mut u8;
    //     unsafe { *ptr = 10; } // Any invalid value for the type
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
    //
    // In order to fix this, we would either need to show that the discriminant computation of
    // `place` is computed in all branches, including the `otherwise` branch, or we would need
    // another analysis pass to determine that the place is fully initialized. It might even be best
    // to have the hoisting be performed in a different pass and just do the CFG changing in this
    // pass.
    for (place, proj) in place.iter_projections() {
        match proj {
            // Dereferencing in the computation of `place` might cause issues from one of two
            // categories. First, the referent might be invalid. We protect against this by
            // dereferencing references only (not pointers). Second, the use of a reference may
            // invalidate other references that are used later (for aliasing reasons). Consider
            // where such an invalidated reference may appear:
            //  - In `Q`: Not possible since `Q` is used as the operand of a `SwitchInt` and so
            //    cannot contain referenced data.
            //  - In `BBU`: Not possible since that block contains only the `unreachable` terminator
            //  - In `BBC.2, BBD.2`: Not possible, since `discriminant(P)` was computed prior to
            //    reaching that block in the input to our transformation, and so any data
            //    invalidated by that computation could not have been used there.
            //  - In `BB9`: Not possible since control flow might have reached `BB9` via the
            //    `otherwise` branch in `BBC, BBD` in the input to our transformation, which would
            //    have invalidated the data when computing `discriminant(P)`
            // So dereferencing here is correct.
            ProjectionElem::Deref => match place.ty(body.local_decls(), tcx).ty.kind() {
                ty::Ref(..) => {}
                _ => return false,
            },
            // Field projections are always valid
            ProjectionElem::Field(..) => {}
            // We cannot allow
            // downcasts either, since the correctness of the downcast may depend on the parent
            // branch being taken. An easy example of this is
            // ```
            // Q = discriminant(_3)
            // P = (_3 as Variant)
            // ```
            // However, checking if the child and parent place are the same and only erroring then
            // is not sufficient either, since the `discriminant(_3) == 1` (or whatever) check may
            // be replaced by another optimization pass with any other condition that can be proven
            // equivalent.
            ProjectionElem::Downcast(..) => {
                return false;
            }
            // We cannot allow indexing since the index may be out of bounds.
            _ => {
                return false;
            }
        }
    }
    true
}

#[derive(Debug)]
struct OptimizationData<'tcx> {
    destination: BasicBlock,
    child_place: Place<'tcx>,
    child_ty: Ty<'tcx>,
    child_source: SourceInfo,
}

fn evaluate_candidate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    parent: BasicBlock,
) -> Option<OptimizationData<'tcx>> {
    let bbs = &body.basic_blocks;
    let TerminatorKind::SwitchInt {
        targets,
        switch_ty: parent_ty,
        ..
    } = &bbs[parent].terminator().kind else {
        return None
    };
    let parent_dest = {
        let poss = targets.otherwise();
        // If the fallthrough on the parent is trivially unreachable, we can let the
        // children choose the destination
        if bbs[poss].statements.len() == 0
            && bbs[poss].terminator().kind == TerminatorKind::Unreachable
        {
            None
        } else {
            Some(poss)
        }
    };
    let (_, child) = targets.iter().next()?;
    let child_terminator = &bbs[child].terminator();
    let TerminatorKind::SwitchInt {
        switch_ty: child_ty,
        targets: child_targets,
        ..
    } = &child_terminator.kind else {
        return None
    };
    if child_ty != parent_ty {
        return None;
    }
    let Some(StatementKind::Assign(boxed))
        = &bbs[child].statements.first().map(|x| &x.kind) else {
        return None;
    };
    let (_, Rvalue::Discriminant(child_place)) = &**boxed else {
        return None;
    };
    let destination = parent_dest.unwrap_or(child_targets.otherwise());

    // Verify that the optimization is legal in general
    // We can hoist evaluating the child discriminant out of the branch
    if !may_hoist(tcx, body, *child_place) {
        return None;
    }

    // Verify that the optimization is legal for each branch
    for (value, child) in targets.iter() {
        if !verify_candidate_branch(&bbs[child], value, *child_place, destination) {
            return None;
        }
    }
    Some(OptimizationData {
        destination,
        child_place: *child_place,
        child_ty: *child_ty,
        child_source: child_terminator.source_info,
    })
}

fn verify_candidate_branch<'tcx>(
    branch: &BasicBlockData<'tcx>,
    value: u128,
    place: Place<'tcx>,
    destination: BasicBlock,
) -> bool {
    // In order for the optimization to be correct, the branch must...
    // ...have exactly one statement
    if branch.statements.len() != 1 {
        return false;
    }
    // ...assign the discriminant of `place` in that statement
    let StatementKind::Assign(boxed) = &branch.statements[0].kind else {
        return false
    };
    let (discr_place, Rvalue::Discriminant(from_place)) = &**boxed else {
        return false
    };
    if *from_place != place {
        return false;
    }
    // ...make that assignment to a local
    if discr_place.projection.len() != 0 {
        return false;
    }
    // ...terminate on a `SwitchInt` that invalidates that local
    let TerminatorKind::SwitchInt{ discr: switch_op, targets, .. } = &branch.terminator().kind else {
        return false
    };
    if *switch_op != Operand::Move(*discr_place) {
        return false;
    }
    // ...fall through to `destination` if the switch misses
    if destination != targets.otherwise() {
        return false;
    }
    // ...have a branch for value `value`
    let mut iter = targets.iter();
    let Some((target_value, _)) = iter.next() else {
        return false;
    };
    if target_value != value {
        return false;
    }
    // ...and have no more branches
    if let Some(_) = iter.next() {
        return false;
    }
    return true;
}
