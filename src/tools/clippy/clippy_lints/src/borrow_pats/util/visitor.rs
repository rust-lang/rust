use std::collections::VecDeque;

use rustc_middle::mir::Body;

use crate::borrow_pats::CfgInfo;

use super::super::prelude::*;
pub trait MyStateInfo<V>: Eq + Clone + std::fmt::Debug {
    fn new(bb: BasicBlock) -> Self;

    /// The return value indicates if the visitor has changed.
    fn join(&mut self, state_owner: &mut V, bb: BasicBlock) -> bool;

    /// This function is called on the input state of a block to compare itself,
    /// to the `con_block` which jumps from within a loop to this state.
    fn check_continue_diff_for_pats(&self, state_owner: &mut V, con_block: BasicBlock);
}

pub trait MyVisitor<'tcx>: Visitor<'tcx> + std::marker::Sized {
    type State: MyStateInfo<Self>;

    fn init_start_block_state(&mut self);

    fn set_state(&mut self, bb: BasicBlock, state: Self::State);
}

pub fn visit_body_with_state<'tcx, V: MyVisitor<'tcx>>(vis: &mut V, info: &AnalysisInfo<'tcx>) {
    let mut stalled_joins = FxHashMap::default();
    for visit in info.visit_order.iter().copied() {
        match visit {
            VisitKind::Next(bb) => {
                // Init state
                if bb == START_BLOCK {
                    vis.init_start_block_state();
                } else {
                    let preds = &info.preds[bb];
                    let mut state = V::State::new(bb);
                    let mut stage_stalled = false;
                    preds.iter().for_each(|bb| {
                        stage_stalled |= !state.join(vis, *bb);
                    });
                    if stage_stalled {
                        stalled_joins.insert(bb, state.clone());
                    }
                    vis.set_state(bb, state);
                }

                // Walk block
                vis.visit_basic_block_data(bb, &info.body.basic_blocks[bb]);
            },
            VisitKind::Continue { from, to } => {
                let state = stalled_joins[&to].clone();
                state.check_continue_diff_for_pats(vis, from);
            },
        }
    }
}

pub fn visit_body<'tcx, V: Visitor<'tcx>>(vis: &mut V, info: &AnalysisInfo<'tcx>) {
    for visit in info.visit_order.iter().copied() {
        if let VisitKind::Next(bb) = visit {
            // Walk block
            vis.visit_basic_block_data(bb, &info.body.basic_blocks[bb]);
        }
    }
}

pub fn unloop_preds<'a>(
    cfg: &'a IndexVec<BasicBlock, CfgInfo>,
    preds: &'a IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>>,
) -> IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>> {
    struct Builder<'a> {
        cfg: &'a IndexVec<BasicBlock, CfgInfo>,
        states: IndexVec<BasicBlock, VisitState>,
        unlooped: IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>>,
    }

    impl<'a> Builder<'a> {
        fn new(
            cfg: &'a IndexVec<BasicBlock, CfgInfo>,
            preds: &'a IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>>,
        ) -> Self {
            let len = cfg.len();
            Self {
                cfg,
                states: IndexVec::from_elem_n(VisitState::Future, len),
                unlooped: preds.clone(),
            }
        }

        fn visit(&mut self, from: BasicBlock, bb: BasicBlock) {
            match self.states[bb] {
                VisitState::Future => {
                    self.states[bb] = VisitState::Current;
                    match &self.cfg[bb] {
                        CfgInfo::Linear(next) => self.visit(bb, *next),
                        CfgInfo::Condition { branches } => {
                            for next in branches {
                                self.visit(bb, *next);
                            }
                        },
                        CfgInfo::Return | CfgInfo::None => {},
                    }

                    self.states[bb] = VisitState::Past;
                },
                VisitState::Current => {
                    self.unlooped[bb].retain(|x| *x != from);
                },
                VisitState::Past => {},
            }
        }
    } // TODO: Check continues

    let mut builder = Builder::new(cfg, preds);
    builder.visit(START_BLOCK, START_BLOCK);
    builder.unlooped
}

#[expect(unused)]
pub fn construct_visit_order(
    body: &Body<'_>,
    cfg: &IndexVec<BasicBlock, CfgInfo>,
    preds: &IndexVec<BasicBlock, SmallVec<[BasicBlock; 1]>>,
) -> Vec<VisitKind> {
    let bb_len = cfg.len();
    let mut visited: BitSet<BasicBlock> = BitSet::new_empty(bb_len);
    let mut order: Vec<VisitKind> = Vec::with_capacity(bb_len);

    let mut queue = VecDeque::new();
    queue.push_back(START_BLOCK);
    while let Some(bb) = queue.pop_front() {
        if visited.contains(bb) {
            continue;
        }

        let preds = &preds[bb];
        if preds.iter().all(|x| visited.contains(*x)) {
            // Not all prerequisites are fulfilled. This bb will be added again by the other branch
            continue;
        }

        match &cfg[bb] {
            CfgInfo::Linear(next) => queue.push_back(*next),
            CfgInfo::Condition { branches } => queue.extend(branches.iter()),
            CfgInfo::None | CfgInfo::Return => {},
        }

        order.push(VisitKind::Next(bb));
        visited.insert(bb);
    }

    order
}

#[derive(Debug, Copy, Clone)]
enum VisitState {
    Future,
    Current,
    Past,
}

#[derive(Debug, Copy, Clone)]
pub enum VisitKind {
    Next(BasicBlock),
    #[expect(unused)]
    Continue {
        from: BasicBlock,
        to: BasicBlock,
    },
}
