use std::iter;

use rustc_abi::Integer;
use rustc_const_eval::const_eval::mk_eval_cx_for_const_val;
use rustc_data_structures::packed::Pu128;
use rustc_index::IndexSlice;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::{IntegerExt, TyAndLayout};
use rustc_middle::ty::util::Discr;
use rustc_middle::ty::{self, ScalarInt, Ty, TyCtxt};

use super::simplify::simplify_cfg;
use crate::patch::MirPatch;

pub(super) struct MatchBranchSimplification;

impl<'tcx> crate::MirPass<'tcx> for MatchBranchSimplification {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let mut apply_patch = false;
        let mut patch = MirPatch::new(body);
        for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
            let Some((discr, targets)) = bb_data.terminator().kind.as_switch() else {
                continue;
            };
            if let Operand::Constant(_) = discr {
                continue;
            }
            let otherwise_is_unreachable =
                body.basic_blocks[targets.otherwise()].is_empty_unreachable();
            let Some((first_case, as_else_case)) =
                candidate_switch(body, bb, targets, otherwise_is_unreachable)
            else {
                continue;
            };

            let discr_ty = discr.ty(body.local_decls(), tcx);
            // Introduce a temporary for the discriminant value.
            let source_info = body.basic_blocks[bb].terminator().source_info;
            let discr_local = patch.new_temp(discr_ty, source_info.span);

            let new_stmts = if body.basic_blocks[first_case].statements.is_empty() {
                Vec::new()
            } else if let Some(else_case) = as_else_case
                && let Some(new_stmts) = simplify_if(
                    tcx,
                    first_case,
                    targets.all_values()[0],
                    else_case,
                    typing_env,
                    &body.basic_blocks,
                    discr_local,
                    discr_ty,
                )
            {
                new_stmts
            } else if otherwise_is_unreachable
                && let Some(new_stmts) = simplify_switch(
                    tcx,
                    body,
                    bb,
                    first_case,
                    targets,
                    typing_env,
                    discr,
                    discr_local,
                    discr_ty,
                )
            {
                new_stmts
            } else {
                patch.revert_last_new_temp();
                continue;
            };

            apply_patch = true;
            // Take ownership of items now that we know we can optimize.
            let discr = discr.clone();

            let (_, first) = targets.iter().next().unwrap();
            let statement_index = body.basic_blocks[bb].statements.len();
            let parent_end = Location { block: bb, statement_index };
            patch.add_statement(parent_end, StatementKind::StorageLive(discr_local));
            patch.add_assign(parent_end, Place::from(discr_local), Rvalue::Use(discr));
            for new_stmt in new_stmts {
                patch.add_statement(parent_end, new_stmt);
            }
            patch.add_statement(parent_end, StatementKind::StorageDead(discr_local));
            patch.patch_terminator(bb, body.basic_blocks[first].terminator().kind.clone());
        }

        if apply_patch {
            patch.apply(body);
            simplify_cfg(tcx, body);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

fn candidate_switch<'tcx, 'a>(
    body: &Body<'tcx>,
    switch_bb: BasicBlock,
    targets: &'a SwitchTargets,
    otherwise_is_unreachable: bool,
) -> Option<(BasicBlock, Option<BasicBlock>)> {
    // We require that the possible target blocks don't contain this block.
    if targets.all_targets().contains(&switch_bb) {
        return None;
    }
    // We require that the possible target blocks all be distinct.
    if !targets.is_distinct() {
        return None;
    }
    let (first_case, other_cases) = match targets.all_targets() {
        &[first, otherwise] => (first, &[otherwise] as &[BasicBlock]),
        &[first, second, _] if otherwise_is_unreachable => (first, &[second] as &[BasicBlock]),
        targets
            if otherwise_is_unreachable
                && let Some((_, cases)) = targets.split_last()
                && let Some((first, others)) = cases.split_first() =>
        {
            (*first, others)
        }
        _ => return None,
    };
    let first_case_bb = &body.basic_blocks[first_case];
    let first_case_terminator_kind = &first_case_bb.terminator().kind;
    let first_case_stmts_len = first_case_bb.statements.len();
    // Check that destinations are identical, and if not, then don't optimize this block
    if !other_cases.iter().all(|&other_case| {
        let other_case_bb = &body.basic_blocks[other_case];
        first_case_stmts_len == other_case_bb.statements.len()
            && first_case_terminator_kind == &other_case_bb.terminator().kind
    }) {
        return None;
    }
    let other_case = if let &[other_case] = other_cases { Some(other_case) } else { None };
    Some((first_case, other_case))
}

/// If a source block is found that switches between two blocks that are exactly
/// the same modulo const bool assignments (e.g., one assigns true another false
/// to the same place), merge a target block statements into the source block,
/// using Eq / Ne comparison with switch value where const bools value differ.
///
/// For example:
///
/// ```ignore (MIR)
/// bb0: {
///     switchInt(move _3) -> [42_isize: bb1, otherwise: bb2];
/// }
///
/// bb1: {
///     _2 = const true;
///     goto -> bb3;
/// }
///
/// bb2: {
///     _2 = const false;
///     goto -> bb3;
/// }
/// ```
///
/// into:
///
/// ```ignore (MIR)
/// bb0: {
///    _2 = Eq(move _3, const 42_isize);
///    goto -> bb3;
/// }
/// ```
fn simplify_if<'tcx>(
    tcx: TyCtxt<'tcx>,
    first_case: BasicBlock,
    first_case_val: Pu128,
    other_case: BasicBlock,
    typing_env: ty::TypingEnv<'tcx>,
    bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
    discr_local: Local,
    discr_ty: Ty<'tcx>,
) -> Option<Vec<StatementKind<'tcx>>> {
    let mut new_stmts = Vec::with_capacity(bbs[first_case].statements.len());
    for (f, s) in iter::zip(&bbs[first_case].statements, &bbs[other_case].statements) {
        match (&f.kind, &s.kind) {
            // If two statements are exactly the same, we can optimize.
            (f_s, s_s) if f_s == s_s => {
                new_stmts.push(f_s.clone());
            }
            // If two statements are const bool assignments to the same place, we can optimize.
            (
                StatementKind::Assign(box (lhs_f, Rvalue::Use(Operand::Constant(f_c)))),
                StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
            ) if lhs_f == lhs_s
                && f_c.const_.ty().is_bool()
                && s_c.const_.ty().is_bool()
                && let Some(first_bool) = f_c.const_.try_eval_bool(tcx, typing_env)
                && let Some(second_bool) = s_c.const_.try_eval_bool(tcx, typing_env) =>
            {
                if first_bool == second_bool {
                    // Same value in both blocks. Use statement as is.
                    new_stmts.push(f.kind.clone());
                } else {
                    // Different value between blocks. Make value conditional on switch
                    // condition.
                    let size = tcx.layout_of(typing_env.as_query_input(discr_ty)).unwrap().size;
                    let const_cmp = Operand::const_from_scalar(
                        tcx,
                        discr_ty,
                        rustc_const_eval::interpret::Scalar::from_uint(first_case_val, size),
                        rustc_span::DUMMY_SP,
                    );
                    let op = if first_bool { BinOp::Eq } else { BinOp::Ne };
                    let rval = Rvalue::BinaryOp(
                        op,
                        Box::new((Operand::Copy(Place::from(discr_local)), const_cmp)),
                    );
                    new_stmts.push(StatementKind::Assign(Box::new((*lhs_f, rval))));
                }
            }
            // Otherwise we cannot optimize. Try another block.
            _ => return None,
        }
    }
    Some(new_stmts)
}

/// Check if the cast constant using `IntToInt` is equal to the target constant.
fn can_cast(
    tcx: TyCtxt<'_>,
    src_val: impl Into<u128>,
    src_layout: TyAndLayout<'_>,
    cast_ty: Ty<'_>,
    target_scalar: ScalarInt,
) -> bool {
    let from_scalar = ScalarInt::try_from_uint(src_val.into(), src_layout.size).unwrap();
    let v = match src_layout.ty.kind() {
        ty::Uint(_) => from_scalar.to_uint(src_layout.size),
        ty::Int(_) => from_scalar.to_int(src_layout.size) as u128,
        // We can also transform the values of other integer representations (such as char),
        // although this may not be practical in real-world scenarios.
        _ => return false,
    };
    let size = match *cast_ty.kind() {
        ty::Int(t) => Integer::from_int_ty(&tcx, t).size(),
        ty::Uint(t) => Integer::from_uint_ty(&tcx, t).size(),
        _ => return false,
    };
    let v = size.truncate(v);
    let cast_scalar = ScalarInt::try_from_uint(v, size).unwrap();
    cast_scalar == target_scalar
}

/// If we find that the value of match is the same as the assignment,
/// merge a target block statements into the source block,
/// using cast to transform different integer types.
///
/// For example:
///
/// ```ignore (MIR)
/// bb0: {
///     switchInt(_1) -> [1: bb2, 2: bb3, 3: bb4, otherwise: bb1];
/// }
///
/// bb1: {
///     unreachable;
/// }
///
/// bb2: {
///     _0 = const 1_i16;
///     goto -> bb5;
/// }
///
/// bb3: {
///     _0 = const 2_i16;
///     goto -> bb5;
/// }
///
/// bb4: {
///     _0 = const 3_i16;
///     goto -> bb5;
/// }
/// ```
///
/// into:
///
/// ```ignore (MIR)
/// bb0: {
///    _0 = _3 as i16 (IntToInt);
///    goto -> bb5;
/// }
/// ```
fn simplify_switch<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    switch_bb: BasicBlock,
    first_case: BasicBlock,
    targets: &SwitchTargets,
    typing_env: ty::TypingEnv<'tcx>,
    discr: &Operand<'tcx>,
    discr_local: Local,
    discr_ty: Ty<'tcx>,
) -> Option<Vec<StatementKind<'tcx>>> {
    if targets.all_values().len() < 2 || targets.all_values().len() > 64 {
        return None;
    }
    // We require that the possible target blocks all be distinct.
    if !targets.is_distinct() {
        return None;
    }
    if !body.basic_blocks[targets.otherwise()].is_empty_unreachable() {
        return None;
    }
    let first_bb: &BasicBlockData<'tcx> = &body.basic_blocks[first_case];

    let mut stmts_iter: Vec<_> =
        targets.iter().map(|(case, bb)| (case, body.basic_blocks[bb].statements.iter())).collect();
    let mut current_line_stmts: Vec<(u128, &StatementKind<'tcx>)> =
        stmts_iter.iter_mut().map(|(case, bb)| (*case, &bb.next().unwrap().kind)).collect();
    let discr_layout = tcx.layout_of(typing_env.as_query_input(discr_ty)).unwrap();
    let mut new_stmts: Vec<StatementKind<'tcx>> = Vec::with_capacity(first_bb.statements.len());
    'finish: loop {
        let new_stmt = simplify_stmt(
            tcx,
            body,
            switch_bb,
            current_line_stmts.as_slice(),
            typing_env,
            discr,
            discr_local,
            discr_ty,
            discr_layout,
        )?;
        new_stmts.push(new_stmt);
        for (current_line_stmt, stmt_iter) in
            iter::zip(current_line_stmts.iter_mut(), stmts_iter.iter_mut())
        {
            let Some(stmt) = stmt_iter.1.next() else {
                break 'finish;
            };
            current_line_stmt.1 = &stmt.kind;
        }
    }
    Some(new_stmts)
}

fn simplify_stmt<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    switch_bb: BasicBlock,
    stmts: &[(u128, &StatementKind<'tcx>)],
    typing_env: ty::TypingEnv<'tcx>,
    discr: &Operand<'tcx>,
    discr_local: Local,
    discr_ty: Ty<'tcx>,
    discr_layout: TyAndLayout<'_>,
) -> Option<StatementKind<'tcx>> {
    let (first_case, first_stmt) = stmts[0];
    // If all statements are exactly the same, we can optimize.
    if stmts.into_iter().skip(1).all(|&(_, stmt)| first_stmt == stmt) {
        return Some(first_stmt.clone());
    }
    let StatementKind::Assign(box (first_lhs, first_rval)) = first_stmt else {
        return None;
    };
    if !stmts.into_iter().skip(1).all(|&(_, stmt)| {
        let StatementKind::Assign(box (other_lhs, _)) = stmt else {
            return false;
        };
        first_lhs == other_lhs
    }) {
        return None;
    }
    if let Rvalue::Use(Operand::Constant(box first_const)) = first_rval
        && first_const.ty().is_integral()
        && let Some(first_scalar_int) = first_const.const_.try_eval_scalar_int(tcx, typing_env)
    {
        if stmts.into_iter().skip(1).all(|&(_, stmt)| {
            let StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(box other_const)))) =
                stmt
            else {
                return false;
            };
            first_const.ty() == other_const.ty()
                && other_const.const_.try_eval_scalar_int(tcx, typing_env) == Some(first_scalar_int)
        }) {
            return Some(first_stmt.clone());
        }

        // Enum variants can also be simplified to an assignment statement,
        // if we can use `IntToInt` cast to get an equal value.
        if can_cast(tcx, first_case, discr_layout, first_const.ty(), first_scalar_int) {
            if stmts.into_iter().skip(1).all(|&(other_case, stmt)| {
                let StatementKind::Assign(box (
                    _,
                    Rvalue::Use(Operand::Constant(box other_const)),
                )) = stmt
                else {
                    return false;
                };
                let Some(other_scalar_int) =
                    other_const.const_.try_eval_scalar_int(tcx, typing_env)
                else {
                    return false;
                };
                first_const.ty() == other_const.ty()
                    && can_cast(tcx, other_case, discr_layout, other_const.ty(), other_scalar_int)
            }) {
                let operand = Operand::Copy(Place::from(discr_local));
                let rval = if first_const.ty() == discr_ty {
                    Rvalue::Use(operand)
                } else {
                    Rvalue::Cast(CastKind::IntToInt, operand, first_const.ty())
                };
                return Some(StatementKind::Assign(Box::new((*first_lhs, rval))));
            }
        }
    }

    if let Some(new_stmt) =
        simplify_to_copy(tcx, body, switch_bb, discr, typing_env, first_lhs, stmts)
    {
        return Some(new_stmt);
    }

    None
}

/// This is primarily used to merge these copy statements that simplified the canonical enum clone method by GVN.
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
/// This function will simplify into a copy statement.
fn simplify_to_copy<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    switch_bb: BasicBlock,
    discr: &Operand<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    first_case_lhs: &Place<'tcx>,
    stmts: &[(u128, &StatementKind<'tcx>)],
) -> Option<StatementKind<'tcx>> {
    let bbs = &body.basic_blocks;
    // Check if the copy source matches the following pattern.
    // _2 = discriminant(*_1); // "*_1" is the expected the copy source.
    // switchInt(move _2) -> [0: bb3, 1: bb2, otherwise: bb1];
    let &Statement {
        kind: StatementKind::Assign(box (discr_place, Rvalue::Discriminant(copy_src_place))),
        ..
    } = bbs[switch_bb].statements.last()?
    else {
        return None;
    };
    if discr.place() != Some(discr_place) {
        return None;
    }
    let src_ty = copy_src_place.ty(body.local_decls(), tcx);
    if !src_ty.ty.is_enum() || src_ty.variant_index.is_some() {
        return None;
    }
    let dest_ty = first_case_lhs.ty(body.local_decls(), tcx);
    if dest_ty.ty != src_ty.ty || dest_ty.variant_index.is_some() {
        return None;
    }
    let ty::Adt(def, _) = dest_ty.ty.kind() else {
        return None;
    };

    for &(case, stmt) in stmts.iter() {
        let StatementKind::Assign(box (_, rvalue)) = stmt else {
            return None;
        };
        match rvalue {
            // Check if `_3 = const Foo::B` can be transformed to `_3 = copy *_1`.
            Rvalue::Use(Operand::Constant(box constant))
                if let Const::Val(const_, ty) = constant.const_ =>
            {
                let (ecx, op) =
                    mk_eval_cx_for_const_val(tcx.at(constant.span), typing_env, const_, ty)?;
                let variant = ecx.read_discriminant(&op).discard_err()?;
                if !def.variants()[variant].fields.is_empty() {
                    return None;
                }
                let Discr { val, .. } = ty.discriminant_for_variant(tcx, variant)?;
                if val != case {
                    return None;
                }
            }
            Rvalue::Use(Operand::Copy(src_place)) if *src_place == copy_src_place => {}
            // Check if `_3 = Foo::B` can be transformed to `_3 = copy *_1`.
            Rvalue::Aggregate(box AggregateKind::Adt(_, variant_index, _, _, None), fields)
                if fields.is_empty()
                    && let Some(Discr { val, .. }) =
                        src_ty.ty.discriminant_for_variant(tcx, *variant_index)
                    && val == case => {}
            _ => return None,
        }
    }
    Some(StatementKind::Assign(Box::new((
        *first_case_lhs,
        Rvalue::Use(Operand::Copy(copy_src_place)),
    ))))
}
