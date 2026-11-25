//! This pass contracts equivalent basic blocks.
//! The equivalence relation is as follows:
//! Exclude the following types of blocks from contracting
//! - is_cleanup is set (these require more invariants, and are on the cold path anyway).
//! - Coverage blocks
//! - The terminator kind is not one of Goto, Drop (no async), or SwitchInt
//! Furthermore, GPU target or the presence of any sanitizer means the pass rejects the whole body.
//!
//! Two eligible blocks are then considered to be equivalent, if
//! - Their statements pairwise match, this includes statement.kind and statement.source_info verbatim.
//! - Their terminators match, again with .kind and .source_info verbatim.

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
            ctx.mir_opt_level() >= 1
                && matches!(ctx.opts.debuginfo, rustc_session::config::DebugInfo::None),
        )
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // do not contract for GPU target.
        if tcx.sess.target.is_like_gpu
            || !tcx.sess.sanitizers().is_empty()
            || body.basic_blocks.len() < 2
        {
            return;
        };

        let classes = EquivalenceClasses::from(body);
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
        are_equivalent(&body.basic_blocks[self.blocks[0]], &body.basic_blocks[bb])
            .then(|| self.blocks.push(bb))
            .is_some()
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

    fn from(body: &Body<'tcx>) -> Self {
        let mut equivalence_classes = EquivalenceClasses::new(body.basic_blocks.len());

        for (bb, bb_d) in body.basic_blocks.iter_enumerated() {
            if fit_for_contracting(bb_d) {
                let mut hasher = FxHasher::default();
                bb_d.statements.len().hash(&mut hasher);
                for statement in bb_d.statements.iter() {
                    std::mem::discriminant(&statement.kind).hash(&mut hasher);
                    statement.source_info.hash(&mut hasher);
                    match &statement.kind {
                        StatementKind::StorageLive(local) => local.hash(&mut hasher),
                        StatementKind::StorageDead(local) => local.hash(&mut hasher),
                        // in derives for large enums, we get a potentially large amount of blocks
                        // that are _x = <...>; goto -> ...
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

                terminator.source_info.hash(&mut hasher);
                for s in terminator.successors() {
                    s.hash(&mut hasher);
                }
                let hash = hasher.finish();
                equivalence_classes.insert(bb, body, hash);
            }
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
    };
    // reject all coverage blocks
    if bb_d.statements.iter().any(|s| matches!(s.kind, StatementKind::Coverage(..))) {
        return false;
    }
    match bb_d.terminator().kind {
        TerminatorKind::Goto { .. } => true,
        TerminatorKind::Drop { drop, .. } => !drop.is_some(),
        TerminatorKind::SwitchInt { .. } => true,
        _ => false,
    }
}

fn are_equivalent<'a>(bb1: &BasicBlockData<'a>, bb2: &BasicBlockData<'a>) -> bool {
    if bb1.statements.len() != bb2.statements.len() {
        return false;
    }

    if bb1
        .statements
        .iter()
        .zip(bb2.statements.iter())
        .any(|(s1, s2)| s1.kind != s2.kind || s1.source_info != s2.source_info)
    {
        return false;
    }

    if bb1.terminator().kind != bb2.terminator().kind
        || bb1.terminator().source_info != bb2.terminator().source_info
    {
        return false;
    }

    true
}
