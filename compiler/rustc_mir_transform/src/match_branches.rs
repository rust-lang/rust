use rustc_index::IndexSlice;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::{ParamEnv, ScalarInt, Ty, TyCtxt};
use rustc_target::abi::Size;
use std::iter;

use super::simplify::simplify_cfg;

pub struct MatchBranchSimplification;

impl<'tcx> MirPass<'tcx> for MatchBranchSimplification {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 1
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let param_env = tcx.param_env_reveal_all_normalized(def_id);

        let mut should_cleanup = false;
        for i in 0..body.basic_blocks.len() {
            let bbs = &*body.basic_blocks;
            let bb_idx = BasicBlock::from_usize(i);
            if !tcx.consider_optimizing(|| format!("MatchBranchSimplification {def_id:?} ")) {
                continue;
            }

            match bbs[bb_idx].terminator().kind {
                TerminatorKind::SwitchInt {
                    discr: ref _discr @ (Operand::Copy(_) | Operand::Move(_)),
                    ref targets,
                    ..
                    // We require that the possible target blocks don't contain this block.
                } if !targets.all_targets().contains(&bb_idx) => {}
                // Only optimize switch int statements
                _ => continue,
            };

            if SimplifyToIf.simplify(tcx, body, bb_idx, param_env).is_some() {
                should_cleanup = true;
                continue;
            }
            // unsound: https://github.com/rust-lang/rust/issues/124150
            if tcx.sess.opts.unstable_opts.unsound_mir_opts
                && SimplifyToExp::default().simplify(tcx, body, bb_idx, param_env).is_some()
            {
                should_cleanup = true;
                continue;
            }
        }

        if should_cleanup {
            simplify_cfg(body);
        }
    }
}

trait SimplifyMatch<'tcx> {
    /// Simplifies a match statement, returning true if the simplification succeeds, false otherwise.
    /// Generic code is written here, and we generally don't need a custom implementation.
    fn simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        body: &mut Body<'tcx>,
        switch_bb_idx: BasicBlock,
        param_env: ParamEnv<'tcx>,
    ) -> Option<()> {
        let bbs = &body.basic_blocks;
        let (discr, targets) = match bbs[switch_bb_idx].terminator().kind {
            TerminatorKind::SwitchInt { ref discr, ref targets, .. } => (discr, targets),
            _ => unreachable!(),
        };

        let discr_ty = discr.ty(body.local_decls(), tcx);
        self.can_simplify(tcx, targets, param_env, bbs, discr_ty)?;

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
        self.new_stmts(tcx, targets, param_env, &mut patch, parent_end, bbs, discr_local, discr_ty);
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
        param_env: ParamEnv<'tcx>,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_ty: Ty<'tcx>,
    ) -> Option<()>;

    fn new_stmts(
        &self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        param_env: ParamEnv<'tcx>,
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
    fn can_simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        param_env: ParamEnv<'tcx>,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        _discr_ty: Ty<'tcx>,
    ) -> Option<()> {
        if targets.iter().len() != 1 {
            return None;
        }
        // We require that the possible target blocks all be distinct.
        let (_, first) = targets.iter().next().unwrap();
        let second = targets.otherwise();
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
                    && f_c.const_.try_eval_bool(tcx, param_env).is_some()
                    && s_c.const_.try_eval_bool(tcx, param_env).is_some() => {}

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
        param_env: ParamEnv<'tcx>,
        patch: &mut MirPatch<'tcx>,
        parent_end: Location,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_local: Local,
        discr_ty: Ty<'tcx>,
    ) {
        let (val, first) = targets.iter().next().unwrap();
        let second = targets.otherwise();
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
                    let f_b = f_c.const_.try_eval_bool(tcx, param_env).unwrap();
                    let s_b = s_c.const_.try_eval_bool(tcx, param_env).unwrap();
                    if f_b == s_b {
                        // Same value in both blocks. Use statement as is.
                        patch.add_statement(parent_end, f.kind.clone());
                    } else {
                        // Different value between blocks. Make value conditional on switch condition.
                        let size = tcx.layout_of(param_env.and(discr_ty)).unwrap().size;
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

#[derive(Default)]
struct SimplifyToExp {
    transfrom_types: Vec<TransfromType>,
}

#[derive(Clone, Copy)]
enum CompareType<'tcx, 'a> {
    /// Identical statements.
    Same(&'a StatementKind<'tcx>),
    /// Assignment statements have the same value.
    Eq(&'a Place<'tcx>, Ty<'tcx>, ScalarInt),
    /// Enum variant comparison type.
    Discr { place: &'a Place<'tcx>, ty: Ty<'tcx>, is_signed: bool },
}

enum TransfromType {
    Same,
    Eq,
    Discr,
}

impl From<CompareType<'_, '_>> for TransfromType {
    fn from(compare_type: CompareType<'_, '_>) -> Self {
        match compare_type {
            CompareType::Same(_) => TransfromType::Same,
            CompareType::Eq(_, _, _) => TransfromType::Eq,
            CompareType::Discr { .. } => TransfromType::Discr,
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
    fn can_simplify(
        &mut self,
        tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        param_env: ParamEnv<'tcx>,
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
        let (first_val, first_target) = target_iter.next().unwrap();
        let first_terminator_kind = &bbs[first_target].terminator().kind;
        // Check that destinations are identical, and if not, then don't optimize this block
        if !targets
            .iter()
            .all(|(_, other_target)| first_terminator_kind == &bbs[other_target].terminator().kind)
        {
            return None;
        }

        let discr_size = tcx.layout_of(param_env.and(discr_ty)).unwrap().size;
        let first_stmts = &bbs[first_target].statements;
        let (second_val, second_target) = target_iter.next().unwrap();
        let second_stmts = &bbs[second_target].statements;
        if first_stmts.len() != second_stmts.len() {
            return None;
        }

        fn int_equal(l: ScalarInt, r: impl Into<u128>, size: Size) -> bool {
            l.to_bits_unchecked() == ScalarInt::try_from_uint(r, size).unwrap().to_bits_unchecked()
        }

        // We first compare the two branches, and then the other branches need to fulfill the same conditions.
        let mut compare_types = Vec::new();
        for (f, s) in iter::zip(first_stmts, second_stmts) {
            let compare_type = match (&f.kind, &s.kind) {
                // If two statements are exactly the same, we can optimize.
                (f_s, s_s) if f_s == s_s => CompareType::Same(f_s),

                // If two statements are assignments with the match values to the same place, we can optimize.
                (
                    StatementKind::Assign(box (lhs_f, Rvalue::Use(Operand::Constant(f_c)))),
                    StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                ) if lhs_f == lhs_s
                    && f_c.const_.ty() == s_c.const_.ty()
                    && f_c.const_.ty().is_integral() =>
                {
                    match (
                        f_c.const_.try_eval_scalar_int(tcx, param_env),
                        s_c.const_.try_eval_scalar_int(tcx, param_env),
                    ) {
                        (Some(f), Some(s)) if f == s => CompareType::Eq(lhs_f, f_c.const_.ty(), f),
                        // Enum variants can also be simplified to an assignment statement if their values are equal.
                        // We need to consider both unsigned and signed scenarios here.
                        (Some(f), Some(s))
                            if ((f_c.const_.ty().is_signed() || discr_ty.is_signed())
                                && int_equal(f, first_val, discr_size)
                                && int_equal(s, second_val, discr_size))
                                || (Some(f) == ScalarInt::try_from_uint(first_val, f.size())
                                    && Some(s)
                                        == ScalarInt::try_from_uint(second_val, s.size())) =>
                        {
                            CompareType::Discr {
                                place: lhs_f,
                                ty: f_c.const_.ty(),
                                is_signed: f_c.const_.ty().is_signed() || discr_ty.is_signed(),
                            }
                        }
                        _ => {
                            return None;
                        }
                    }
                }

                // Otherwise we cannot optimize. Try another block.
                _ => return None,
            };
            compare_types.push(compare_type);
        }

        // All remaining BBs need to fulfill the same pattern as the two BBs from the previous step.
        for (other_val, other_target) in target_iter {
            let other_stmts = &bbs[other_target].statements;
            if compare_types.len() != other_stmts.len() {
                return None;
            }
            for (f, s) in iter::zip(&compare_types, other_stmts) {
                match (*f, &s.kind) {
                    (CompareType::Same(f_s), s_s) if f_s == s_s => {}
                    (
                        CompareType::Eq(lhs_f, f_ty, val),
                        StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                    ) if lhs_f == lhs_s
                        && s_c.const_.ty() == f_ty
                        && s_c.const_.try_eval_scalar_int(tcx, param_env) == Some(val) => {}
                    (
                        CompareType::Discr { place: lhs_f, ty: f_ty, is_signed },
                        StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                    ) if lhs_f == lhs_s && s_c.const_.ty() == f_ty => {
                        let Some(f) = s_c.const_.try_eval_scalar_int(tcx, param_env) else {
                            return None;
                        };
                        if is_signed
                            && s_c.const_.ty().is_signed()
                            && int_equal(f, other_val, discr_size)
                        {
                            continue;
                        }
                        if Some(f) == ScalarInt::try_from_uint(other_val, f.size()) {
                            continue;
                        }
                        return None;
                    }
                    _ => return None,
                }
            }
        }
        self.transfrom_types = compare_types.into_iter().map(|c| c.into()).collect();
        Some(())
    }

    fn new_stmts(
        &self,
        _tcx: TyCtxt<'tcx>,
        targets: &SwitchTargets,
        _param_env: ParamEnv<'tcx>,
        patch: &mut MirPatch<'tcx>,
        parent_end: Location,
        bbs: &IndexSlice<BasicBlock, BasicBlockData<'tcx>>,
        discr_local: Local,
        discr_ty: Ty<'tcx>,
    ) {
        let (_, first) = targets.iter().next().unwrap();
        let first = &bbs[first];

        for (t, s) in iter::zip(&self.transfrom_types, &first.statements) {
            match (t, &s.kind) {
                (TransfromType::Same, _) | (TransfromType::Eq, _) => {
                    patch.add_statement(parent_end, s.kind.clone());
                }
                (
                    TransfromType::Discr,
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
