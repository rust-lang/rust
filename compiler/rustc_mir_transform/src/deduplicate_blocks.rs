//! This pass finds basic blocks that are completely equal,
//! and replaces all uses with just one of them.

use std::{collections::hash_map::Entry, hash::Hash, hash::Hasher, iter};

use crate::MirPass;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use super::simplify::simplify_cfg;

pub struct DeduplicateBlocks;

impl<'tcx> MirPass<'tcx> for DeduplicateBlocks {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.mir_opt_level() < 4 {
            return;
        }
        debug!("Running DeduplicateBlocks on `{:?}`", body.source);
        let duplicates = find_duplicates(body);
        let has_opts_to_apply = !duplicates.is_empty();

        if has_opts_to_apply {
            let mut opt_applier = OptApplier { tcx, duplicates };
            opt_applier.visit_body(body);
            simplify_cfg(tcx, body);
        }
    }
}

struct OptApplier<'tcx> {
    tcx: TyCtxt<'tcx>,
    duplicates: FxHashMap<BasicBlock, BasicBlock>,
}

impl<'tcx> MutVisitor<'tcx> for OptApplier<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_terminator(&mut self, terminator: &mut Terminator<'tcx>, location: Location) {
        for target in terminator.successors_mut() {
            if let Some(replacement) = self.duplicates.get(target) {
                debug!("SUCCESS: Replacing: `{:?}` with `{:?}`", target, replacement);
                *target = *replacement;
            }
        }

        self.super_terminator(terminator, location);
    }
}

fn find_duplicates<'a, 'tcx>(body: &'a Body<'tcx>) -> FxHashMap<BasicBlock, BasicBlock> {
    let mut duplicates = FxHashMap::default();

    let bbs_to_go_through =
        body.basic_blocks().iter_enumerated().filter(|(_, bbd)| !bbd.is_cleanup).count();

    let mut same_hashes =
        FxHashMap::with_capacity_and_hasher(bbs_to_go_through, Default::default());

    // Go through the basic blocks backwards. This means that in case of duplicates,
    // we can use the basic block with the highest index as the replacement for all lower ones.
    // For example, if bb1, bb2 and bb3 are duplicates, we will first insert bb3 in same_hashes.
    // Then we will see that bb2 is a duplicate of bb3,
    // and insert bb2 with the replacement bb3 in the duplicates list.
    // When we see bb1, we see that it is a duplicate of bb3, and therefore insert it in the duplicates list
    // with replacement bb3.
    // When the duplicates are removed, we will end up with only bb3.
    for (bb, bbd) in body.basic_blocks().iter_enumerated().rev().filter(|(_, bbd)| !bbd.is_cleanup)
    {
        // Basic blocks can get really big, so to avoid checking for duplicates in basic blocks
        // that are unlikely to have duplicates, we stop early. The early bail number has been
        // found experimentally by eprintln while compiling the crates in the rustc-perf suite.
        if bbd.statements.len() > 10 {
            continue;
        }

        let to_hash = BasicBlockHashable { basic_block_data: bbd };
        let entry = same_hashes.entry(to_hash);
        match entry {
            Entry::Occupied(occupied) => {
                // The basic block was already in the hashmap, which means we have a duplicate
                let value = *occupied.get();
                debug!("Inserting {:?} -> {:?}", bb, value);
                duplicates.try_insert(bb, value).expect("key was already inserted");
            }
            Entry::Vacant(vacant) => {
                vacant.insert(bb);
            }
        }
    }

    duplicates
}

struct BasicBlockHashable<'tcx, 'a> {
    basic_block_data: &'a BasicBlockData<'tcx>,
}

impl<'tcx, 'a> Hash for BasicBlockHashable<'tcx, 'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_statements(state, self.basic_block_data.statements.iter());
        // Note that since we only hash the kind, we lose span information if we deduplicate the blocks
        self.basic_block_data.terminator().kind.hash(state);
    }
}

impl<'tcx, 'a> Eq for BasicBlockHashable<'tcx, 'a> {}

impl<'tcx, 'a> PartialEq for BasicBlockHashable<'tcx, 'a> {
    fn eq(&self, other: &Self) -> bool {
        self.basic_block_data.statements.len() == other.basic_block_data.statements.len()
            && &self.basic_block_data.terminator().kind == &other.basic_block_data.terminator().kind
            && iter::zip(&self.basic_block_data.statements, &other.basic_block_data.statements)
                .all(|(x, y)| statement_eq(&x.kind, &y.kind))
    }
}

fn hash_statements<'a, 'tcx, H: Hasher>(
    hasher: &mut H,
    iter: impl Iterator<Item = &'a Statement<'tcx>>,
) where
    'tcx: 'a,
{
    for stmt in iter {
        statement_hash(hasher, &stmt.kind);
    }
}

fn statement_hash<'tcx, H: Hasher>(hasher: &mut H, stmt: &StatementKind<'tcx>) {
    match stmt {
        StatementKind::Assign(box (place, rvalue)) => {
            place.hash(hasher);
            rvalue_hash(hasher, rvalue)
        }
        x => x.hash(hasher),
    };
}

fn rvalue_hash<H: Hasher>(hasher: &mut H, rvalue: &Rvalue<'tcx>) {
    match rvalue {
        Rvalue::Use(op) => operand_hash(hasher, op),
        x => x.hash(hasher),
    };
}

fn operand_hash<H: Hasher>(hasher: &mut H, operand: &Operand<'tcx>) {
    match operand {
        Operand::Constant(box Constant { user_ty: _, literal, span: _ }) => literal.hash(hasher),
        x => x.hash(hasher),
    };
}

fn statement_eq<'tcx>(lhs: &StatementKind<'tcx>, rhs: &StatementKind<'tcx>) -> bool {
    let res = match (lhs, rhs) {
        (
            StatementKind::Assign(box (place, rvalue)),
            StatementKind::Assign(box (place2, rvalue2)),
        ) => place == place2 && rvalue_eq(rvalue, rvalue2),
        (x, y) => x == y,
    };
    debug!("statement_eq lhs: `{:?}` rhs: `{:?}` result: {:?}", lhs, rhs, res);
    res
}

fn rvalue_eq(lhs: &Rvalue<'tcx>, rhs: &Rvalue<'tcx>) -> bool {
    let res = match (lhs, rhs) {
        (Rvalue::Use(op1), Rvalue::Use(op2)) => operand_eq(op1, op2),
        (x, y) => x == y,
    };
    debug!("rvalue_eq lhs: `{:?}` rhs: `{:?}` result: {:?}", lhs, rhs, res);
    res
}

fn operand_eq(lhs: &Operand<'tcx>, rhs: &Operand<'tcx>) -> bool {
    let res = match (lhs, rhs) {
        (
            Operand::Constant(box Constant { user_ty: _, literal, span: _ }),
            Operand::Constant(box Constant { user_ty: _, literal: literal2, span: _ }),
        ) => literal == literal2,
        (x, y) => x == y,
    };
    debug!("operand_eq lhs: `{:?}` rhs: `{:?}` result: {:?}", lhs, rhs, res);
    res
}
