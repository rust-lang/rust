//! This pass attempts to merge all branches to eliminate switch terminator.
//! Ideally, we could combine it with `MatchBranchSimplification`, as these two passes
//! match and merge statements with different patterns. Given the compile time and
//! code complexity, we have not merged them into a more general pass for now.
use rustc_const_eval::const_eval::mk_eval_cx_for_const_val;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty;
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{ParamEnv, TyCtxt};
use rustc_mir_dataflow::impls::borrowed_locals;

use crate::dead_store_elimination::DeadStoreAnalysis;

pub(super) struct MergeBranchSimplification;

impl<'tcx> crate::MirPass<'tcx> for MergeBranchSimplification {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let param_env = tcx.param_env_reveal_all_normalized(def_id);

        let borrowed_locals = borrowed_locals(body);
        let mut dead_store_analysis = DeadStoreAnalysis::new(tcx, body, &borrowed_locals);

        for switch_bb_idx in body.basic_blocks.indices() {
            let bbs = &*body.basic_blocks;
            let Some((switch_discr, targets)) = bbs[switch_bb_idx].terminator().kind.as_switch()
            else {
                continue;
            };
            // Check that destinations are identical, and if not, then don't optimize this block.
            let mut targets_iter = targets.iter();
            let first_terminator_kind = &bbs[targets_iter.next().unwrap().1].terminator().kind;
            if targets_iter.any(|(_, other_target)| {
                first_terminator_kind != &bbs[other_target].terminator().kind
            }) {
                continue;
            }
            // We require that the possible target blocks all be distinct.
            if !targets.is_distinct() {
                continue;
            }
            if !bbs[targets.otherwise()].is_empty_unreachable() {
                continue;
            }
            // Check if the copy source matches the following pattern.
            // _2 = discriminant(*_1); // "*_1" is the expected the copy source.
            // switchInt(move _2) -> [0: bb3, 1: bb2, otherwise: bb1];
            let Some(&Statement {
                kind: StatementKind::Assign(box (discr_place, Rvalue::Discriminant(src_place))),
                ..
            }) = bbs[switch_bb_idx].statements.last()
            else {
                continue;
            };
            if switch_discr.place() != Some(discr_place) {
                continue;
            }
            let src_ty = src_place.ty(body.local_decls(), tcx);
            if let Some(dest_place) = can_simplify_to_copy(
                tcx,
                param_env,
                body,
                targets,
                src_place,
                src_ty,
                &mut dead_store_analysis,
            ) {
                let statement_index = bbs[switch_bb_idx].statements.len();
                let parent_end = Location { block: switch_bb_idx, statement_index };
                let mut patch = MirPatch::new(body);
                patch.add_assign(parent_end, dest_place, Rvalue::Use(Operand::Copy(src_place)));
                patch.patch_terminator(switch_bb_idx, first_terminator_kind.clone());
                patch.apply(body);
                super::simplify::remove_dead_blocks(body);
                // After modifying the MIR, the result of `MaybeTransitiveLiveLocals` may become invalid,
                // keeping it simple to process only once.
                break;
            }
        }
    }
}

/// The GVN simplified
/// ```ignore (syntax-highlighting-only)
/// match a {
///     Foo::A(x) => Foo::A(*x),
///     Foo::B => Foo::B
/// }
/// ```
/// to
/// ```ignore (syntax-highlighting-only)
/// match a {
///     Foo::A(_x) => a, // copy a
///     Foo::B => Foo::B
/// }
/// ```
/// This function answers whether it can be simplified to a copy statement
/// by returning the copy destination.
fn can_simplify_to_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    body: &Body<'tcx>,
    targets: &SwitchTargets,
    src_place: Place<'tcx>,
    src_ty: tcx::PlaceTy<'tcx>,
    dead_store_analysis: &mut DeadStoreAnalysis<'tcx, '_, '_>,
) -> Option<Place<'tcx>> {
    let mut targets_iter = targets.iter();
    let (first_index, first_target) = targets_iter.next()?;
    let dest_place = find_copy_assign(
        tcx,
        param_env,
        body,
        first_index,
        first_target,
        src_place,
        src_ty,
        dead_store_analysis,
    )?;
    let dest_ty = dest_place.ty(body.local_decls(), tcx);
    if dest_ty.ty != src_ty.ty {
        return None;
    }
    for (other_index, other_target) in targets_iter {
        if dest_place
            != find_copy_assign(
                tcx,
                param_env,
                body,
                other_index,
                other_target,
                src_place,
                src_ty,
                dead_store_analysis,
            )?
        {
            return None;
        }
    }
    Some(dest_place)
}

// Find the single assignment statement where the source of the copy is from the source.
// All other statements are dead statements or have no effect that can be eliminated.
fn find_copy_assign<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ParamEnv<'tcx>,
    body: &Body<'tcx>,
    index: u128,
    target_block: BasicBlock,
    src_place: Place<'tcx>,
    src_ty: tcx::PlaceTy<'tcx>,
    dead_store_analysis: &mut DeadStoreAnalysis<'tcx, '_, '_>,
) -> Option<Place<'tcx>> {
    let statements = &body.basic_blocks[target_block].statements;
    if statements.is_empty() {
        return None;
    }
    let assign_stmt = if statements.len() == 1 {
        0
    } else {
        let mut lived_stmts: BitSet<usize> = BitSet::new_filled(statements.len());
        let mut expected_assign_stmt = None;
        for (statement_index, statement) in statements.iter().enumerate().rev() {
            let loc = Location { block: target_block, statement_index };
            if dead_store_analysis.is_dead_store(loc, &statement.kind) {
                lived_stmts.remove(statement_index);
            } else if matches!(
                statement.kind,
                StatementKind::StorageLive(_) | StatementKind::StorageDead(_)
            ) {
            } else if matches!(statement.kind, StatementKind::Assign(_))
                && expected_assign_stmt.is_none()
            {
                // There is only one assign statement that cannot be ignored
                // that can be used as an expected copy statement.
                expected_assign_stmt = Some(statement_index);
                lived_stmts.remove(statement_index);
            } else {
                return None;
            }
        }
        let expected_assign = expected_assign_stmt?;
        if !lived_stmts.is_empty() {
            // We can ignore the paired StorageLive and StorageDead.
            let mut storage_live_locals: BitSet<Local> = BitSet::new_empty(body.local_decls.len());
            for stmt_index in lived_stmts.iter() {
                let statement = &statements[stmt_index];
                match &statement.kind {
                    StatementKind::StorageLive(local) if storage_live_locals.insert(*local) => {}
                    StatementKind::StorageDead(local) if storage_live_locals.remove(*local) => {}
                    _ => return None,
                }
            }
            if !storage_live_locals.is_empty() {
                return None;
            }
        }
        expected_assign
    };
    let &(dest_place, ref rvalue) = statements[assign_stmt].kind.as_assign()?;
    let dest_ty = dest_place.ty(body.local_decls(), tcx);
    if dest_ty.ty != src_ty.ty {
        return None;
    }
    let ty::Adt(def, _) = dest_ty.ty.kind() else {
        return None;
    };
    match rvalue {
        // Check if `_3 = const Foo::B` can be transformed to `_3 = copy *_1`.
        Rvalue::Use(Operand::Constant(box constant))
            if let Const::Val(const_, ty) = constant.const_ =>
        {
            let (ecx, op) = mk_eval_cx_for_const_val(tcx.at(constant.span), param_env, const_, ty)?;
            let variant = ecx.read_discriminant(&op).discard_err()?;
            if !def.variants()[variant].fields.is_empty() {
                return None;
            }
            let Discr { val, .. } = ty.discriminant_for_variant(tcx, variant)?;
            if val != index {
                return None;
            }
        }
        Rvalue::Use(Operand::Copy(place)) if *place == src_place => {}
        // Check if `_3 = Foo::B` can be transformed to `_3 = copy *_1`.
        Rvalue::Aggregate(box AggregateKind::Adt(_, variant_index, _, _, None), fields)
            if fields.is_empty()
                && let Some(Discr { val, .. }) =
                    src_ty.ty.discriminant_for_variant(tcx, *variant_index)
                && val == index => {}
        _ => return None,
    }
    Some(dest_place)
}
