use rustc_abi::Integer;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::{IntegerExt, TyAndLayout};
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
        let mut changed = false;
        for bb in body.basic_blocks.indices() {
            if !candidate_match(body, bb) {
                continue;
            };
            changed |= simplify_match(tcx, typing_env, body, bb)
        }

        if changed {
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
    patch: MirPatch<'tcx>,
    body: &'a Body<'tcx>,
    switch_bb: BasicBlock,
    discr_local: Option<Local>,
    discr_ty: Ty<'tcx>,
}

impl<'tcx, 'a> SimplifyMatch<'tcx, 'a> {
    fn discr_local(&mut self) -> Local {
        *self.discr_local.get_or_insert_with(|| {
            // Introduce a temporary for the discriminant value.
            let source_info = self.body.basic_blocks[self.switch_bb].terminator().source_info;
            self.patch.new_temp(self.discr_ty, source_info.span)
        })
    }

    /// Merges the assignments if all rvalues are constants and equal.
    fn merge_if_equal_const(
        &self,
        dest: Place<'tcx>,
        consts: &[(u128, &ConstOperand<'tcx>)],
        otherwise: Option<&ConstOperand<'tcx>>,
    ) -> Option<StatementKind<'tcx>> {
        let (_, first_const, mut others) = split_first_case(consts, otherwise);
        let first_scalar_int = first_const.const_.try_eval_scalar_int(self.tcx, self.typing_env)?;
        if others.all(|const_| {
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
    fn merge_by_eq_op(
        &mut self,
        dest: Place<'tcx>,
        consts: &[(u128, &ConstOperand<'tcx>)],
        otherwise: Option<&ConstOperand<'tcx>>,
    ) -> Option<StatementKind<'tcx>> {
        // FIXME: extend to any case.
        let (first_case, first_const, mut others) = split_first_case(consts, otherwise);
        if !first_const.ty().is_bool() {
            return None;
        }
        let first_bool = first_const.const_.try_eval_bool(self.tcx, self.typing_env)?;
        if others.all(|const_| {
            const_.const_.try_eval_bool(self.tcx, self.typing_env) == Some(!first_bool)
        }) {
            // Make value conditional on switch condition.
            let size =
                self.tcx.layout_of(self.typing_env.as_query_input(self.discr_ty)).unwrap().size;
            let const_cmp = Operand::const_from_scalar(
                self.tcx,
                self.discr_ty,
                rustc_const_eval::interpret::Scalar::from_uint(first_case, size),
                rustc_span::DUMMY_SP,
            );
            let op = if first_bool { BinOp::Eq } else { BinOp::Ne };
            let rval = Rvalue::BinaryOp(
                op,
                Box::new((Operand::Copy(Place::from(self.discr_local())), const_cmp)),
            );
            Some(StatementKind::Assign(Box::new((dest, rval))))
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
        &mut self,
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
            let operand = Operand::Copy(Place::from(self.discr_local()));
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

    /// Returns a new statement if we can use the statement replace all statements.
    fn try_merge_stmts(
        &mut self,
        _index: usize,
        stmts: &[(u128, &StatementKind<'tcx>)],
        otherwise: Option<&StatementKind<'tcx>>,
    ) -> Option<StatementKind<'tcx>> {
        if let Some(new_stmt) = identical_stmts(stmts, otherwise) {
            return Some(new_stmt);
        }

        let (dest, rvals, otherwise) = candidate_assign(stmts, otherwise)?;
        if let Some((consts, otherwise)) = candidate_const(&rvals, otherwise) {
            if let Some(new_stmt) = self.merge_if_equal_const(dest, &consts, otherwise) {
                return Some(new_stmt);
            }
            if let Some(new_stmt) = self.merge_by_eq_op(dest, &consts, otherwise) {
                return Some(new_stmt);
            }
            // Requires the otherwise is unreachable.
            if otherwise.is_none()
                && let Some(new_stmt) = self.merge_by_int_to_int(dest, &consts)
            {
                return Some(new_stmt);
            }
        }
        None
    }
}

/// Returns the first case target if all targets have an equal number of statements and identical destination.
fn candidate_match<'tcx>(body: &Body<'tcx>, switch_bb: BasicBlock) -> bool {
    let targets = match &body.basic_blocks[switch_bb].terminator().kind {
        TerminatorKind::SwitchInt {
            discr: Operand::Copy(_) | Operand::Move(_), targets, ..
        } => targets,
        // Only optimize switch int statements
        _ => return false,
    };
    // We require that the possible target blocks don't contain this block.
    if targets.all_targets().contains(&switch_bb) {
        return false;
    }
    // We require that the possible target blocks all be distinct.
    if !targets.is_distinct() {
        return false;
    }
    let &[first, ref others @ .., otherwise] = targets.all_targets() else {
        return false;
    };
    let first_case_bb = &body.basic_blocks[first];
    let first_case_terminator_kind = &first_case_bb.terminator().kind;
    let first_case_stmts_len = first_case_bb.statements.len();

    let otherwise =
        if body.basic_blocks[otherwise].is_empty_unreachable() { None } else { Some(&otherwise) };
    // Check that destinations are identical, and if not, then don't optimize this block
    others.iter().chain(otherwise).all(|&bb| {
        let bb = &body.basic_blocks[bb];
        first_case_stmts_len == bb.statements.len()
            && first_case_terminator_kind == &bb.terminator().kind
    })
}

fn simplify_match<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    body: &mut Body<'tcx>,
    switch_bb: BasicBlock,
) -> bool {
    let (discr, targets) = match &body.basic_blocks[switch_bb].terminator().kind {
        TerminatorKind::SwitchInt { discr, targets, .. } => (discr, targets),
        _ => unreachable!(),
    };
    let mut simplify_match = SimplifyMatch {
        tcx,
        typing_env,
        patch: MirPatch::new(body),
        body,
        switch_bb,
        discr_local: None,
        discr_ty: discr.ty(body.local_decls(), tcx),
    };
    let stmts: Vec<_> = targets
        .iter()
        .map(|(case, bb)| (case, simplify_match.body.basic_blocks[bb].statements.as_slice()))
        .collect();
    let mut new_stmts = Vec::new();
    let otherwise_stmts = if body.basic_blocks[targets.otherwise()].is_empty_unreachable() {
        None
    } else {
        Some(body.basic_blocks[targets.otherwise()].statements.as_slice())
    };
    let first_case_bb = targets.all_targets()[0];
    let stmt_len = body.basic_blocks[first_case_bb].statements.len();
    let mut cases = Vec::with_capacity(stmt_len);
    // Check at each position in the basic blocks whether these statements can be merged.
    for index in 0..stmt_len {
        let otherwise = otherwise_stmts.map(|stmt| &stmt[index].kind);
        cases.clear();
        for &(case, stmts) in &stmts {
            cases.push((case, &stmts[index].kind));
        }
        let Some(new_stmt) = simplify_match.try_merge_stmts(index, cases.as_slice(), otherwise)
        else {
            return false;
        };
        new_stmts.push(new_stmt);
    }
    // Take ownership of items now that we know we can optimize.
    let discr = discr.clone();

    let statement_index = body.basic_blocks[switch_bb].statements.len();
    let parent_end = Location { block: switch_bb, statement_index };
    let mut patch = simplify_match.patch;
    if let Some(discr_local) = simplify_match.discr_local {
        patch.add_statement(parent_end, StatementKind::StorageLive(discr_local));
        patch.add_assign(parent_end, Place::from(discr_local), Rvalue::Use(discr));
    }
    for new_stmt in new_stmts {
        patch.add_statement(parent_end, new_stmt);
    }
    if let Some(discr_local) = simplify_match.discr_local {
        patch.add_statement(parent_end, StatementKind::StorageDead(discr_local));
    }
    patch.patch_terminator(switch_bb, body.basic_blocks[first_case_bb].terminator().kind.clone());
    patch.apply(body);
    true
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
    otherwise: Option<&'a StatementKind<'tcx>>,
) -> Option<(Place<'tcx>, Vec<(u128, &'a Rvalue<'tcx>)>, Option<&'a Rvalue<'tcx>>)> {
    let (_, first_stmt) = stmts[0];
    let (dest, _) = first_stmt.as_assign()?;
    let otherwise = if let Some(otherwise) = otherwise {
        let Some((otherwise_dest, rval)) = otherwise.as_assign() else {
            return None;
        };
        if otherwise_dest != dest {
            return None;
        }
        Some(rval)
    } else {
        None
    };
    let rvals = stmts
        .into_iter()
        .map(|&(case, stmt)| {
            let (other_dest, rval) = stmt.as_assign()?;
            if other_dest != dest {
                return None;
            }
            Some((case, rval))
        })
        .try_collect()?;
    Some((*dest, rvals, otherwise))
}

// Returns all ConstOperands if all Rvalues are ConstOperands.
fn candidate_const<'tcx, 'a>(
    rvals: &'a [(u128, &'a Rvalue<'tcx>)],
    otherwise: Option<&'a Rvalue<'tcx>>,
) -> Option<(Vec<(u128, &'a ConstOperand<'tcx>)>, Option<&'a ConstOperand<'tcx>>)> {
    let otherwise = if let Some(otherwise) = otherwise {
        let Rvalue::Use(Operand::Constant(box const_)) = otherwise else {
            return None;
        };
        Some(const_)
    } else {
        None
    };
    let consts = rvals
        .into_iter()
        .map(|&(case, rval)| {
            let Rvalue::Use(Operand::Constant(box const_)) = rval else { return None };
            Some((case, const_))
        })
        .try_collect()?;
    Some((consts, otherwise))
}

// Returns the first case and others (including otherwise if present).
fn split_first_case<'a, T>(
    stmts: &'a [(u128, &'a T)],
    otherwise: Option<&'a T>,
) -> (u128, &'a T, impl Iterator<Item = &'a T>) {
    let (first_case, first) = stmts[0];
    (first_case, first, stmts[1..].into_iter().map(|&(_, val)| val).chain(otherwise))
}

// If all statements are identical, we can optimize.
fn identical_stmts<'tcx>(
    stmts: &[(u128, &StatementKind<'tcx>)],
    otherwise: Option<&StatementKind<'tcx>>,
) -> Option<StatementKind<'tcx>> {
    use itertools::Itertools;
    let (_, first_stmt, others) = split_first_case(stmts, otherwise);
    if std::iter::once(first_stmt).chain(others).all_equal() {
        return Some(first_stmt.clone());
    }
    None
}
