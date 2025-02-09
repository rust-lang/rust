#![allow(unused_imports, unused_variables, dead_code)]
//! This pass removes unneded temporary locals.
//! Eg. This:
//! ```
//! _3 = copy _1;
//! _4 = Add(_3, _2);
//! ```
//! Will get turned into this:
//! ```
//! _4 = Add(_1,_2)
//! ```
use rustc_index::IndexVec;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::{debug, trace};

pub(super) struct PropTrivialLocals;
struct StatmentPos(BasicBlock, usize);
impl StatmentPos {
    fn nop_out(self, blocks: &mut IndexVec<BasicBlock, BasicBlockData<'_>>) {
        blocks[self.0].statements[self.1].make_nop();
    }
}
fn propagate_operand<'tcx>(
    operand: &mut Operand<'tcx>,
    local: Local,
    local_replacement: &Operand<'tcx>,
) -> bool {
    let place = match operand {
        Operand::Copy(place) | Operand::Move(place) => place,
        _ => return false,
    };
    if place.local != local {
        return false;
    }
    if place.projection.is_empty() {
        *operand = local_replacement.clone();
        return true;
    }
    return false;
}
impl<'tcx> crate::MirPass<'tcx> for PropTrivialLocals {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 1
    }
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running PropTrivialLocals on {:?}", body.source);
        loop {
            let mut dead_candidates = false;
            for (bid, block) in body.basic_blocks.as_mut().iter_enumerated_mut() {
                let mut iter = StatementPairIterator::new(bid, &mut block.statements);
                while let Some(StatementPair((a_idx, a), (b_idx, b))) = iter.next() {
                    let (StatementKind::Assign(tmp_a), StatementKind::Assign(tmp_b)) =
                        (&a.kind, &mut b.kind)
                    else {
                        continue;
                    };
                    let Some(loc_a) = tmp_a.0.as_local() else {
                        continue;
                    };
                    let Rvalue::Use(ref src_a) = tmp_a.1 else {
                        continue;
                    };
                    match &mut tmp_b.1 {
                        Rvalue::BinaryOp(_, args) => {
                            dead_candidates |= propagate_operand(&mut args.0, loc_a, src_a)
                                || propagate_operand(&mut args.1, loc_a, src_a);
                        }
                        // I could add a `| Rvalue::Use(arg)` here, but I have not encountered this pattern in compiler-generated MIR *ever*,
                        // so it is likely not worth even checking for. Likevise, `Rvalue::Ref | Rvalue::RawPtr` also seems to never benfit from this opt.
                        Rvalue::UnaryOp(_, arg)
                        | Rvalue::Cast(_, arg, _)
                        | Rvalue::Repeat(arg, _) => {
                            dead_candidates |= propagate_operand(arg, loc_a, src_a);
                        }
                        Rvalue::Aggregate(_, args) => {
                            dead_candidates |=
                                args.iter_mut().any(|arg| propagate_operand(arg, loc_a, src_a))
                        }
                        _ => continue,
                    }
                }
            }
            if dead_candidates {
                crate::simplify::remove_unused_definitions(body);
            } else {
                break;
            }
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}
struct DeadLocalCandidates {
    dead: Local,
}
struct StatementPair<'a, 'tcx>(
    (StatmentPos, &'a mut Statement<'tcx>),
    (StatmentPos, &'a mut Statement<'tcx>),
);
struct StatementPairIterator<'a, 'tcx> {
    curr_block: BasicBlock,
    curr_idx: usize,
    statements: &'a mut [Statement<'tcx>],
}
impl<'a, 'tcx> StatementPairIterator<'a, 'tcx> {
    fn new(curr_block: BasicBlock, statements: &'a mut [Statement<'tcx>]) -> Self {
        Self { curr_block, statements, curr_idx: 0 }
    }
    fn get_at<'s>(&'s mut self, idx_a: usize, idx_b: usize) -> StatementPair<'s, 'tcx> {
        // Some bounds checks
        assert!(idx_a < idx_b);
        assert!(idx_b < self.statements.len());
        let (part_a, reminder) = self.statements.split_at_mut(idx_a + 1);
        let (part_b, reminder) = reminder.split_at_mut((idx_b - part_a.len()) + 1);
        let a = &mut part_a[part_a.len() - 1];
        let b = &mut part_b[part_b.len() - 1];
        StatementPair(
            (StatmentPos(self.curr_block, idx_a), a),
            (StatmentPos(self.curr_block, idx_b), b),
        )
    }
    fn is_statement_irrelevant(statement: &Statement<'_>) -> bool {
        match statement.kind {
            StatementKind::Nop
            | StatementKind::StorageLive(..)
            | StatementKind::PlaceMention(..)
            | StatementKind::Coverage(..)
            | StatementKind::ConstEvalCounter => true,
            StatementKind::Assign(..)
            | StatementKind::FakeRead(..)
            | StatementKind::SetDiscriminant { .. }
            | StatementKind::Deinit(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Retag(..)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::BackwardIncompatibleDropHint { .. } => false,
        }
    }
    fn next<'s>(&'s mut self) -> Option<StatementPair<'s, 'tcx>> {
        // Skip irrelevant statements
        if self.curr_idx >= self.statements.len() {
            return None;
        }
        while Self::is_statement_irrelevant(&self.statements[self.curr_idx]) {
            self.curr_idx += 1;
            if self.curr_idx >= self.statements.len() {
                return None;
            }
        }
        let curr = self.curr_idx;
        self.curr_idx += 1;
        let mut next_idx = self.curr_idx;
        if next_idx >= self.statements.len() {
            return None;
        }
        while Self::is_statement_irrelevant(&self.statements[next_idx]) {
            next_idx += 1;
            if next_idx >= self.statements.len() {
                return None;
            }
        }
        Some(self.get_at(curr, next_idx))
    }
}
