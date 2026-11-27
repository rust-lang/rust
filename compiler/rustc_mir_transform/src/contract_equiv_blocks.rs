//! This pass contracts equivalent basic blocks.
//!
//! Exclude the following types of blocks from contracting
//! - is_cleanup is set (these require more invariants, and are on the cold path anyway).
//! - Any statement is not one of StorageLive/Dead, Nop or Assign
//! - There are more than 8 statements
//! - The terminator kind is not one of Goto, Drop (no async), or SwitchInt
//! - It is unreachable
//! Furthermore, GPU target or the presence of any sanitizer means the pass rejects the whole body.
//!
//! Two eligible blocks are then considered to be equivalent, if
//! - Their statement.kind's match.
//! - Their terminator.kind's match.
//! - Each successor must dominate both or neither of the blocks
//! The last rule avoids introducing irreducible control flow
//!
//! Enabled by default at mir_opt_level >= 2 and with debuginfo off
//! Contracting blocks will destroy debuginfo

use std::hash::{Hash, Hasher};

use rustc_data_structures::fx::{FxHashMap, FxHasher};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use smallvec::{SmallVec, smallvec};

use crate::PassPolicy;

pub(super) struct ContractEquivBlocks;

impl<'tcx> crate::MirPass<'tcx> for ContractEquivBlocks {
    fn policy(&self, ctx: &crate::PassCtx<'_>) -> PassPolicy {
        PassPolicy::optional(
            ctx.mir_opt_level() >= 2
                && matches!(ctx.opts.debuginfo, rustc_session::config::DebugInfo::None),
        )
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.target.is_like_gpu
            || !tcx.sess.sanitizers().is_empty()
            || body.basic_blocks.len() < 2
        {
            return;
        };

        let classes = EquivalenceClasses::construct_classes(body);
        contract_classes(body, classes);
    }
}

struct EquivalenceClass {
    // a vector of basic blocks that are equivalent
    blocks: SmallVec<[BasicBlock; 2]>,
}

impl EquivalenceClass {
    fn new(bb: BasicBlock) -> Self {
        EquivalenceClass { blocks: smallvec![bb] }
    }

    fn try_push<'tcx>(&mut self, bb: BasicBlock, body: &Body<'tcx>) -> bool {
        are_equivalent(self.blocks[0], bb, body).then(|| self.blocks.push(bb)).is_some()
    }

    fn len(&self) -> usize {
        self.blocks.len()
    }
}

struct EquivalenceClasses {
    // a blocks hash points to a vector of equivalence classes
    // each equivalence class is a vector of basic blocks
    map: FxHashMap<u64, SmallVec<[EquivalenceClass; 1]>>,
}

impl<'tcx> EquivalenceClasses {
    fn new(capacity: usize) -> Self {
        EquivalenceClasses {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    fn insert(&mut self, bb: BasicBlock, body: &Body<'tcx>, hash: u64) {
        if let Some(classes) = self.map.get_mut(&hash) {
            for class in classes.iter_mut() {
                if class.try_push(bb, body) {
                    return;
                }
            }
            // if no equivalent class was found, create a new one
            let new_class = EquivalenceClass::new(bb);
            classes.push(new_class);
        } else {
            // if no classes exist for this hash, create a new one
            let new_class = EquivalenceClass::new(bb);
            self.map.insert(hash, smallvec![new_class]);
        }
    }

    fn construct_classes(body: &Body<'tcx>) -> Self {
        let mut equivalence_classes = EquivalenceClasses::new(body.basic_blocks.len());

        for (bb, bb_d) in body.basic_blocks.iter_enumerated() {
            if !fit_for_contracting(bb_d) || !body.basic_blocks.dominators().is_reachable(bb) {
                continue;
            }
            let mut hasher = FxHasher::default();
            bb_d.statements.len().hash(&mut hasher);
            for statement in bb_d.statements.iter() {
                std::mem::discriminant(&statement.kind).hash(&mut hasher);
                match &statement.kind {
                    StatementKind::StorageLive(local) => local.hash(&mut hasher),
                    StatementKind::StorageDead(local) => local.hash(&mut hasher),
                    // in derives for large enums or large match tables, we get a potentially large amount of
                    // blocks that are _x = <...>; goto -> ...
                    // These are distinguished by the following. Otherwise they would be
                    // distinguished in the exact check later which is expensive
                    StatementKind::Assign(assign) => {
                        let (_place, rvalue) = &**assign;
                        if let Rvalue::Use(Operand::Constant(c), _) = rvalue {
                            c.const_.hash(&mut hasher);
                        }
                    }
                    _ => (),
                }
            }

            let terminator = bb_d.terminator();
            std::mem::discriminant(&terminator.kind).hash(&mut hasher);

            for s in terminator.successors() {
                s.hash(&mut hasher);
            }
            let hash = hasher.finish();
            equivalence_classes.insert(bb, body, hash);
        }
        equivalence_classes
    }
}

fn contract_classes<'tcx>(body: &mut Body<'tcx>, classes: EquivalenceClasses) {
    // unstable query results is fine with .all
    #[allow(rustc::potential_query_instability)]
    if classes.map.values().all(|classes| classes.iter().all(|class| class.len() == 1)) {
        return;
    };
    let blocks = body.basic_blocks.as_mut();
    // allow unstable because the order of the classes does not affect the final result
    #[allow(rustc::potential_query_instability)]
    for e_classes in classes.map.values() {
        for e_class in e_classes {
            if e_class.len() == 1 {
                continue;
            };
            // link all blocks in a class to the last block. Last block, not first, otherwise we
            // might link to the START_BLOCK
            let last_block_i = *e_class.blocks.last().unwrap(); // Equivalence classes are not empty
            for bb in e_class.blocks.iter().rev().skip(1) {
                blocks[*bb].statements.clear();
                blocks[*bb].terminator_mut().kind = TerminatorKind::goto(last_block_i);
            }
        }
    }
}

fn fit_for_contracting<'tcx>(bb_d: &BasicBlockData<'tcx>) -> bool {
    //reject cleanup as they are on the cold path anyway and need more invariants
    if bb_d.is_cleanup {
        return false;
    }
    if bb_d.statements.len() > 8 {
        return false;
    }
    for statement in bb_d.statements.iter() {
        match statement.kind {
            StatementKind::StorageLive { .. } => (),
            StatementKind::StorageDead { .. } => (),
            StatementKind::Nop => (),
            StatementKind::Assign { .. } => (),
            _ => return false,
        }
    }
    match bb_d.terminator().kind {
        TerminatorKind::Goto { .. } => true,
        TerminatorKind::Drop { drop, .. } => !drop.is_some(),
        TerminatorKind::SwitchInt { .. } => true,
        _ => false,
    }
}

fn are_equivalent<'tcx>(bb1: BasicBlock, bb2: BasicBlock, body: &Body<'tcx>) -> bool {
    let bb1_d = &body.basic_blocks[bb1];
    let bb2_d = &body.basic_blocks[bb2];
    if bb1_d.statements.len() != bb2_d.statements.len() {
        return false;
    }

    if bb1_d.statements.iter().zip(bb2_d.statements.iter()).any(|(s1, s2)| s1.kind != s2.kind) {
        return false;
    }

    if bb1_d.terminator().kind != bb2_d.terminator().kind {
        return false;
    }

    let dominators = body.basic_blocks.dominators();

    for s in bb1_d.terminator().successors() {
        // we filter out unreachable blocks earlier, so dominates() cannot panic here
        let s_dom_bb1 = dominators.dominates(s, bb1);
        let s_dom_bb2 = dominators.dominates(s, bb2);
        if s_dom_bb1 != s_dom_bb2 {
            return false;
        }
    }

    true
}
