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

/// Merges all targets into one basic block if each statement can have the same statement.
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
            let Some((first_case, as_else_target)) =
                candidate_switch(body, bb, targets, otherwise_is_unreachable)
            else {
                continue;
            };

            let discr_ty = discr.ty(body.local_decls(), tcx);
            // Introduce a temporary for the discriminant value.
            let source_info = body.basic_blocks[bb].terminator().source_info;
            let discr_local = patch.new_temp(discr_ty, source_info.span);

            let simplify_match = SimplifyMatch {
                tcx,
                typing_env,
                body,
                switch_bb: bb,
                discr,
                discr_local,
                discr_ty,
            };

            let new_stmts = if body.basic_blocks[first_case].statements.is_empty() {
                Vec::new()
            } else if let Some(else_target) = as_else_target
                && let Some(new_stmts) = simplify_if(
                    tcx,
                    first_case,
                    targets.all_values()[0],
                    else_target,
                    typing_env,
                    &body.basic_blocks,
                    discr_local,
                    discr_ty,
                )
            {
                new_stmts
            } else if otherwise_is_unreachable
                && let Some(new_stmts) = simplify_match.simplify_switch(targets)
            {
                new_stmts
            } else {
                patch.revert_last_new_temp();
                continue;
            };

            apply_patch = true;
            // Take ownership of items now that we know we can optimize.
            let discr = discr.clone();

            let statement_index = body.basic_blocks[bb].statements.len();
            let parent_end = Location { block: bb, statement_index };
            patch.add_statement(parent_end, StatementKind::StorageLive(discr_local));
            patch.add_assign(parent_end, Place::from(discr_local), Rvalue::Use(discr));
            for new_stmt in new_stmts {
                patch.add_statement(parent_end, new_stmt);
            }
            patch.add_statement(parent_end, StatementKind::StorageDead(discr_local));
            patch.patch_terminator(bb, body.basic_blocks[first_case].terminator().kind.clone());
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

struct SimplifyMatch<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    body: &'a Body<'tcx>,
    switch_bb: BasicBlock,
    discr: &'a Operand<'tcx>,
    discr_local: Local,
    discr_ty: Ty<'tcx>,
}

impl<'tcx, 'a> SimplifyMatch<'tcx, 'a> {
    /// Merges the assignments if all rvalues are constants and equal.
    fn merge_if_equal_const(
        &self,
        dest: Place<'tcx>,
        consts: &[(u128, &ConstOperand<'tcx>)],
    ) -> Option<StatementKind<'tcx>> {
        let (_, first_const) = consts[0];
        let first_scalar_int = first_const.const_.try_eval_scalar_int(self.tcx, self.typing_env)?;
        if consts.iter().skip(1).all(|&(_, const_)| {
            const_.const_.try_eval_scalar_int(self.tcx, self.typing_env) == Some(first_scalar_int)
        }) {
            Some(StatementKind::Assign(Box::new((
                dest,
                Rvalue::Use(Operand::Constant(Box::new(first_const.clone()))),
            ))))
        } else {
            None
        }
    }

    /// Merges the assignments if all rvalues can be cast from the discriminant value by IntToInt.
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
    ///    _0 = _1 as i16 (IntToInt);
    ///    goto -> bb5;
    /// }
    /// ```
    fn merge_by_int_to_int(
        &self,
        dest: Place<'tcx>,
        consts: &[(u128, &ConstOperand<'tcx>)],
    ) -> Option<StatementKind<'tcx>> {
        let (_, first_const) = consts[0];
        if !first_const.ty().is_integral() {
            return None;
        }
        let discr_layout =
            self.tcx.layout_of(self.typing_env.as_query_input(self.discr_ty)).unwrap();
        if consts.iter().all(|&(case, const_)| {
            let Some(scalar_int) = const_.const_.try_eval_scalar_int(self.tcx, self.typing_env)
            else {
                return false;
            };
            can_cast(self.tcx, case, discr_layout, const_.ty(), scalar_int)
        }) {
            let operand = Operand::Copy(Place::from(self.discr_local));
            let rval = if first_const.ty() == self.discr_ty {
                Rvalue::Use(operand)
            } else {
                Rvalue::Cast(CastKind::IntToInt, operand, first_const.ty())
            };
            Some(StatementKind::Assign(Box::new((dest, rval))))
        } else {
            None
        }
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
    /// This will simplify into a copy statement.
    fn merge_by_copy(
        &self,
        dest: Place<'tcx>,
        rvals: &[(u128, &Rvalue<'tcx>)],
    ) -> Option<StatementKind<'tcx>> {
        let bbs = &self.body.basic_blocks;
        // Check if the copy source matches the following pattern.
        // _2 = discriminant(*_1); // "*_1" is the expected the copy source.
        // switchInt(move _2) -> [0: bb3, 1: bb2, otherwise: bb1];
        let &Statement {
            kind: StatementKind::Assign(box (discr_place, Rvalue::Discriminant(copy_src_place))),
            ..
        } = bbs[self.switch_bb].statements.last()?
        else {
            return None;
        };
        if self.discr.place() != Some(discr_place) {
            return None;
        }
        let src_ty = copy_src_place.ty(self.body.local_decls(), self.tcx);
        if !src_ty.ty.is_enum() || src_ty.variant_index.is_some() {
            return None;
        }
        let dest_ty = dest.ty(self.body.local_decls(), self.tcx);
        if dest_ty.ty != src_ty.ty || dest_ty.variant_index.is_some() {
            return None;
        }
        let ty::Adt(def, _) = dest_ty.ty.kind() else {
            return None;
        };

        for &(case, rvalue) in rvals.iter() {
            match rvalue {
                // Check if `_3 = const Foo::B` can be transformed to `_3 = copy *_1`.
                Rvalue::Use(Operand::Constant(box constant))
                    if let Const::Val(const_, ty) = constant.const_ =>
                {
                    let (ecx, op) = mk_eval_cx_for_const_val(
                        self.tcx.at(constant.span),
                        self.typing_env,
                        const_,
                        ty,
                    )?;
                    let variant = ecx.read_discriminant(&op).discard_err()?;
                    if !def.variants()[variant].fields.is_empty() {
                        return None;
                    }
                    let Discr { val, .. } = ty.discriminant_for_variant(self.tcx, variant)?;
                    if val != case {
                        return None;
                    }
                }
                Rvalue::Use(Operand::Copy(src_place)) if *src_place == copy_src_place => {}
                // Check if `_3 = Foo::B` can be transformed to `_3 = copy *_1`.
                Rvalue::Aggregate(box AggregateKind::Adt(_, variant_index, _, _, None), fields)
                    if fields.is_empty()
                        && let Some(Discr { val, .. }) =
                            src_ty.ty.discriminant_for_variant(self.tcx, *variant_index)
                        && val == case => {}
                _ => return None,
            }
        }
        Some(StatementKind::Assign(Box::new((dest, Rvalue::Use(Operand::Copy(copy_src_place))))))
    }

    fn try_merge_stmts(
        &self,
        stmts: &[(u128, &StatementKind<'tcx>)],
    ) -> Option<StatementKind<'tcx>> {
        if let Some(new_stmt) = identical_stmts(stmts) {
            return Some(new_stmt);
        }

        let (dest, rvals) = candidate_assign(stmts)?;
        if let Some(consts) = candidate_const(&rvals) {
            if let Some(new_stmt) = self.merge_if_equal_const(dest, &consts) {
                return Some(new_stmt);
            }
            if let Some(new_stmt) = self.merge_by_int_to_int(dest, &consts) {
                return Some(new_stmt);
            }
        }

        if let Some(new_stmt) = self.merge_by_copy(dest, &rvals) {
            return Some(new_stmt);
        }
        None
    }

    fn simplify_switch(&self, targets: &SwitchTargets) -> Option<Vec<StatementKind<'tcx>>> {
        let mut stmts_iter: Vec<_> = targets
            .iter()
            .map(|(case, bb)| (case, self.body.basic_blocks[bb].statements.iter()))
            .collect();
        let mut current_line_stmts: Vec<(u128, &StatementKind<'tcx>)> =
            stmts_iter.iter_mut().map(|(case, bb)| (*case, &bb.next().unwrap().kind)).collect();
        let mut new_stmts = Vec::new();
        'finish: loop {
            let new_stmt = self.try_merge_stmts(current_line_stmts.as_slice())?;
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
}

/// Returns the first case target and the else target if all targets have an equal number of statements and identical destination.
/// The else target has value only if this can be regarded if branch.
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

fn candidate_assign<'tcx, 'a>(
    stmts: &'a [(u128, &'a StatementKind<'tcx>)],
) -> Option<(Place<'tcx>, Vec<(u128, &'a Rvalue<'tcx>)>)> {
    let (_, first_stmt) = stmts[0];
    let (dest, _) = first_stmt.as_assign()?;
    if !stmts
        .into_iter()
        .skip(1)
        .all(|&(_, stmt)| stmt.as_assign().map(|(dest, _)| dest) == Some(dest))
    {
        return None;
    }
    let rvals = stmts
        .into_iter()
        .map(|&(case, stmt)| {
            let Some((_, rval)) = stmt.as_assign() else {
                unreachable!();
            };
            (case, rval)
        })
        .collect();
    Some((*dest, rvals))
}

// Returns all ConstOperands if all Rvalues are ConstOperands.
fn candidate_const<'tcx, 'a>(
    rvals: &'a [(u128, &'a Rvalue<'tcx>)],
) -> Option<Vec<(u128, &'a ConstOperand<'tcx>)>> {
    if rvals.into_iter().all(|(_, rval)| {
        if let Rvalue::Use(Operand::Constant(box _)) = rval {
            return true;
        }
        false
    }) {
        Some(
            rvals
                .into_iter()
                .map(|&(case, rval)| {
                    let Rvalue::Use(Operand::Constant(box const_)) = rval else {
                        unreachable!();
                    };
                    (case, const_)
                })
                .collect(),
        )
    } else {
        None
    }
}

// If all statements are identical, we can optimize.
fn identical_stmts<'tcx>(stmts: &[(u128, &StatementKind<'tcx>)]) -> Option<StatementKind<'tcx>> {
    let (_, first_stmt) = stmts[0];
    if stmts.into_iter().skip(1).all(|&(_, stmt)| first_stmt == stmt) {
        return Some(first_stmt.clone());
    }
    None
}
