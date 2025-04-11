use std::iter;

use rustc_abi::Integer;
use rustc_index::IndexSlice;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::{IntegerExt, TyAndLayout};
use rustc_middle::ty::{self, ScalarInt, Ty, TyCtxt};
use tracing::instrument;

use super::simplify::simplify_cfg;
use crate::patch::MirPatch;

pub(super) struct MatchBranchSimplification;

impl<'tcx> crate::MirPass<'tcx> for MatchBranchSimplification {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let typing_env = body.typing_env(tcx);
        let mut should_cleanup = false;
        for bb_idx in body.basic_blocks.indices() {
            match &body.basic_blocks[bb_idx].terminator().kind {
                TerminatorKind::SwitchInt {
                    discr: Operand::Copy(_) | Operand::Move(_),
                    targets,
                    ..
                    // We require that the possible target blocks don't contain this block.
                } if !targets.all_targets().contains(&bb_idx) => {}
                // Only optimize switch int statements
                _ => continue,
            };

            if SimplifyToIf.simplify(tcx, body, bb_idx, typing_env).is_some() {
                should_cleanup = true;
                continue;
            }
            if SimplifyToExp::default().simplify(tcx, body, bb_idx, typing_env).is_some() {
                should_cleanup = true;
                continue;
            }
        }

        if should_cleanup {
            simplify_cfg(body);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

trait SimplifyMatch<'tcx> {
    /// Simplifies a match statement, returning `Some` if the simplification succeeds, `None`
    /// otherwise. Generic code is written here, and we generally don't need a custom
    /// implementation.
    fn simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        body: &mut Body<'tcx>,
        switch_bb_idx: BasicBlock,
        typing_env: ty::TypingEnv<'tcx>,
    ) -> Option<()> {
        let bbs = &body.basic_blocks;
        let TerminatorKind::SwitchInt { discr, targets, .. } =
            &bbs[switch_bb_idx].terminator().kind
        else {
            unreachable!();
        };

        let discr_ty = discr.ty(body.local_decls(), tcx);
        self.can_simplify(tcx, targets, typing_env, bbs, discr_ty)?;

        let mut patch = MirPatch::new(body);

        // Take ownership of items now that we know we can optimize.
        let discr = discr.clone();

        // Introduce a temporary for the discriminant value.
        let source_info = bbs[switch_bb_idx].terminator().source_info;
        let discr_local = patch.new_temp(discr_ty, source_info.span);

        let (_, first) = targets.iter().next().unwrap();
        let statement_index = bbs[switch_bb_idx].statements.len();
        let parent_end = Location { block: switch_bb_idx, statement_index };
        patch.add_statement(parent_end, StatementKind::StorageLive(discr_local));
        patch.add_assign(parent_end, Place::from(discr_local), Rvalue::Use(discr));
        self.new_stmts(
            tcx,
            targets,
            typing_env,
            &mut patch,
            parent_end,
            bbs,
            discr_local,
            discr_ty,
        );
        patch.add_statement(parent_end, StatementKind::StorageDead(discr_local));
        patch.patch_terminator(switch_bb_idx, bbs[first].terminator().kind.clone());
        patch.apply(body);
        Some(())
    }

    /// Check that the BBs to be simplified satisfies all distinct and
    /// that the terminator are the same.
    /// There are also conditions for different ways of simplification.
    fn can_simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        typing_env: ty::TypingEnv<'tcx>,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_ty: Ty<'tcx>,
    ) -> Option<()>;

    fn new_stmts(
        &self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        typing_env: ty::TypingEnv<'tcx>,
        patch: &mut MirPatch<'tcx>,
        parent_end: Location,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_local: Local,
        discr_ty: Ty<'tcx>,
    );
}

struct SimplifyToIf;

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
impl<'tcx> SimplifyMatch<'tcx> for SimplifyToIf {
    #[instrument(level = "debug", skip(self, tcx), ret)]
    fn can_simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        typing_env: ty::TypingEnv<'tcx>,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        _discr_ty: Ty<'tcx>,
    ) -> Option<()> {
        let (first, second) = match targets.all_targets() {
            &[first, otherwise] => (first, otherwise),
            &[first, second, otherwise] if bbs[otherwise].is_empty_unreachable() => (first, second),
            _ => {
                return None;
            }
        };

        // We require that the possible target blocks all be distinct.
        if first == second {
            return None;
        }
        // Check that destinations are identical, and if not, then don't optimize this block
        if bbs[first].terminator().kind != bbs[second].terminator().kind {
            return None;
        }

        // Check that blocks are assignments of consts to the same place or same statement,
        // and match up 1-1, if not don't optimize this block.
        let first_stmts = &bbs[first].statements;
        let second_stmts = &bbs[second].statements;
        if first_stmts.len() != second_stmts.len() {
            return None;
        }
        for (f, s) in iter::zip(first_stmts, second_stmts) {
            match (&f.kind, &s.kind) {
                // If two statements are exactly the same, we can optimize.
                (f_s, s_s) if f_s == s_s => {}

                // If two statements are const bool assignments to the same place, we can optimize.
                (
                    StatementKind::Assign(box (lhs_f, Rvalue::Use(Operand::Constant(f_c)))),
                    StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                ) if lhs_f == lhs_s
                    && f_c.const_.ty().is_bool()
                    && s_c.const_.ty().is_bool()
                    && f_c.const_.try_eval_bool(tcx, typing_env).is_some()
                    && s_c.const_.try_eval_bool(tcx, typing_env).is_some() => {}

                // Otherwise we cannot optimize. Try another block.
                _ => return None,
            }
        }
        Some(())
    }

    fn new_stmts(
        &self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        typing_env: ty::TypingEnv<'tcx>,
        patch: &mut MirPatch<'tcx>,
        parent_end: Location,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_local: Local,
        discr_ty: Ty<'tcx>,
    ) {
        let ((val, first), second) = match (targets.all_targets(), targets.all_values()) {
            (&[first, otherwise], &[val]) => ((val, first), otherwise),
            (&[first, second, otherwise], &[val, _]) if bbs[otherwise].is_empty_unreachable() => {
                ((val, first), second)
            }
            _ => unreachable!(),
        };

        // We already checked that first and second are different blocks,
        // and bb_idx has a different terminator from both of them.
        let first = &bbs[first];
        let second = &bbs[second];
        for (f, s) in iter::zip(&first.statements, &second.statements) {
            match (&f.kind, &s.kind) {
                (f_s, s_s) if f_s == s_s => {
                    patch.add_statement(parent_end, f.kind.clone());
                }

                (
                    StatementKind::Assign(box (lhs, Rvalue::Use(Operand::Constant(f_c)))),
                    StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(s_c)))),
                ) => {
                    // From earlier loop we know that we are dealing with bool constants only:
                    let f_b = f_c.const_.try_eval_bool(tcx, typing_env).unwrap();
                    let s_b = s_c.const_.try_eval_bool(tcx, typing_env).unwrap();
                    if f_b == s_b {
                        // Same value in both blocks. Use statement as is.
                        patch.add_statement(parent_end, f.kind.clone());
                    } else {
                        // Different value between blocks. Make value conditional on switch
                        // condition.
                        let size = tcx.layout_of(typing_env.as_query_input(discr_ty)).unwrap().size;
                        let const_cmp = Operand::const_from_scalar(
                            tcx,
                            discr_ty,
                            rustc_const_eval::interpret::Scalar::from_uint(val, size),
                            rustc_span::DUMMY_SP,
                        );
                        let op = if f_b { BinOp::Eq } else { BinOp::Ne };
                        let rhs = Rvalue::BinaryOp(
                            op,
                            Box::new((Operand::Copy(Place::from(discr_local)), const_cmp)),
                        );
                        patch.add_assign(parent_end, *lhs, rhs);
                    }
                }

                _ => unreachable!(),
            }
        }
    }
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
        _ => unreachable!("invalid int"),
    };
    let size = match *cast_ty.kind() {
        ty::Int(t) => Integer::from_int_ty(&tcx, t).size(),
        ty::Uint(t) => Integer::from_uint_ty(&tcx, t).size(),
        _ => unreachable!("invalid int"),
    };
    let v = size.truncate(v);
    let cast_scalar = ScalarInt::try_from_uint(v, size).unwrap();
    cast_scalar == target_scalar
}

#[derive(Default)]
struct SimplifyToExp {
    transform_kinds: Vec<TransformKind>,
}

#[derive(Clone, Copy, Debug)]
enum ExpectedTransformKind<'a, 'tcx> {
    /// Identical statements.
    Same(&'a StatementKind<'tcx>),
    /// Assignment statements have the same value.
    SameByEq { place: &'a Place<'tcx>, ty: Ty<'tcx>, scalar: ScalarInt },
    /// Enum variant comparison type.
    Cast { place: &'a Place<'tcx>, ty: Ty<'tcx> },
}

enum TransformKind {
    Same,
    Cast,
}

impl From<ExpectedTransformKind<'_, '_>> for TransformKind {
    fn from(compare_type: ExpectedTransformKind<'_, '_>) -> Self {
        match compare_type {
            ExpectedTransformKind::Same(_) => TransformKind::Same,
            ExpectedTransformKind::SameByEq { .. } => TransformKind::Same,
            ExpectedTransformKind::Cast { .. } => TransformKind::Cast,
        }
    }
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
impl<'tcx> SimplifyMatch<'tcx> for SimplifyToExp {
    #[instrument(level = "debug", skip(self, tcx), ret)]
    fn can_simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        typing_env: ty::TypingEnv<'tcx>,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_ty: Ty<'tcx>,
    ) -> Option<()> {
        if targets.iter().len() < 2 || targets.iter().len() > 64 {
            return None;
        }
        // We require that the possible target blocks all be distinct.
        if !targets.is_distinct() {
            return None;
        }
        if !bbs[targets.otherwise()].is_empty_unreachable() {
            return None;
        }
        let mut target_iter = targets.iter();
        let (first_case_val, first_target) = target_iter.next().unwrap();
        let first_terminator_kind = &bbs[first_target].terminator().kind;
        // Check that destinations are identical, and if not, then don't optimize this block
        if !targets
            .iter()
            .all(|(_, other_target)| first_terminator_kind == &bbs[other_target].terminator().kind)
        {
            return None;
        }

        let discr_layout = tcx.layout_of(typing_env.as_query_input(discr_ty)).unwrap();
        let first_stmts = &bbs[first_target].statements;
        let (second_case_val, second_target) = target_iter.next().unwrap();
        let second_stmts = &bbs[second_target].statements;
        if first_stmts.len() != second_stmts.len() {
            return None;
        }

        // We first compare the two branches, and then the other branches need to fulfill the same
        // conditions.
        let mut expected_transform_kinds = Vec::new();
        for (f, s) in iter::zip(first_stmts, second_stmts) {
            let compare_type = match (&f.kind, &s.kind) {
                // If two statements are exactly the same, we can optimize.
                (f_s, s_s) if f_s == s_s => ExpectedTransformKind::Same(f_s),

                // If two statements are assignments with the match values to the same place, we
                // can optimize.
                (
                    StatementKind::Assign(box (lhs_f, Rvalue::Use(Operand::Constant(f_c)))),
                    StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                ) if lhs_f == lhs_s
                    && f_c.const_.ty() == s_c.const_.ty()
                    && f_c.const_.ty().is_integral() =>
                {
                    match (
                        f_c.const_.try_eval_scalar_int(tcx, typing_env),
                        s_c.const_.try_eval_scalar_int(tcx, typing_env),
                    ) {
                        (Some(f), Some(s)) if f == s => ExpectedTransformKind::SameByEq {
                            place: lhs_f,
                            ty: f_c.const_.ty(),
                            scalar: f,
                        },
                        // Enum variants can also be simplified to an assignment statement,
                        // if we can use `IntToInt` cast to get an equal value.
                        (Some(f), Some(s))
                            if (can_cast(
                                tcx,
                                first_case_val,
                                discr_layout,
                                f_c.const_.ty(),
                                f,
                            ) && can_cast(
                                tcx,
                                second_case_val,
                                discr_layout,
                                f_c.const_.ty(),
                                s,
                            )) =>
                        {
                            ExpectedTransformKind::Cast { place: lhs_f, ty: f_c.const_.ty() }
                        }
                        _ => {
                            return None;
                        }
                    }
                }

                // Otherwise we cannot optimize. Try another block.
                _ => return None,
            };
            expected_transform_kinds.push(compare_type);
        }

        // All remaining BBs need to fulfill the same pattern as the two BBs from the previous step.
        for (other_val, other_target) in target_iter {
            let other_stmts = &bbs[other_target].statements;
            if expected_transform_kinds.len() != other_stmts.len() {
                return None;
            }
            for (f, s) in iter::zip(&expected_transform_kinds, other_stmts) {
                match (*f, &s.kind) {
                    (ExpectedTransformKind::Same(f_s), s_s) if f_s == s_s => {}
                    (
                        ExpectedTransformKind::SameByEq { place: lhs_f, ty: f_ty, scalar },
                        StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                    ) if lhs_f == lhs_s
                        && s_c.const_.ty() == f_ty
                        && s_c.const_.try_eval_scalar_int(tcx, typing_env) == Some(scalar) => {}
                    (
                        ExpectedTransformKind::Cast { place: lhs_f, ty: f_ty },
                        StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                    ) if let Some(f) = s_c.const_.try_eval_scalar_int(tcx, typing_env)
                        && lhs_f == lhs_s
                        && s_c.const_.ty() == f_ty
                        && can_cast(tcx, other_val, discr_layout, f_ty, f) => {}
                    _ => return None,
                }
            }
        }
        self.transform_kinds = expected_transform_kinds.into_iter().map(|c| c.into()).collect();
        Some(())
    }

    fn new_stmts(
        &self,
        _tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        _typing_env: ty::TypingEnv<'tcx>,
        patch: &mut MirPatch<'tcx>,
        parent_end: Location,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_local: Local,
        discr_ty: Ty<'tcx>,
    ) {
        let (_, first) = targets.iter().next().unwrap();
        let first = &bbs[first];

        for (t, s) in iter::zip(&self.transform_kinds, &first.statements) {
            match (t, &s.kind) {
                (TransformKind::Same, _) => {
                    patch.add_statement(parent_end, s.kind.clone());
                }
                (
                    TransformKind::Cast,
                    StatementKind::Assign(box (lhs, Rvalue::Use(Operand::Constant(f_c)))),
                ) => {
                    let operand = Operand::Copy(Place::from(discr_local));
                    let r_val = if f_c.const_.ty() == discr_ty {
                        Rvalue::Use(operand)
                    } else {
                        Rvalue::Cast(CastKind::IntToInt, operand, f_c.const_.ty())
                    };
                    patch.add_assign(parent_end, *lhs, r_val);
                }
                _ => unreachable!(),
            }
        }
    }
}
