use std::hash::Hash;
use std::ops::Range;

use derive_where::derive_where;
use rustc_index::IndexVec;
use rustc_type_ir::data_structures::{HashMap, HashSet};

use crate::search_graph::{AvailableDepth, Cx, CycleHeads, PathKind, Stack, StackDepth};

#[derive_where(Debug, Clone, Copy; X: Cx)]
pub(super) struct GoalInfo<X: Cx> {
    pub input: X::Input,
    pub step_kind_from_parent: PathKind,
    pub available_depth: AvailableDepth,
}

rustc_index::newtype_index! {
    #[orderable]
    #[gate_rustc_only]
    pub struct NodeId {} // TODO: private
}

rustc_index::newtype_index! {
    #[orderable]
    #[gate_rustc_only]
    pub(super) struct CycleId {}
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub(super) enum RebaseEntriesKind {
    Normal,
    Ambiguity,
    Overflow,
}

#[derive_where(Debug; X: Cx)]
pub(super) enum NodeKind<X: Cx> {
    InProgress {
        cycles_start: CycleId,
        last_iteration_provisional_result: Option<X::Result>,
        rebase_entries_kind: Option<RebaseEntriesKind>,
    },
    Finished {
        encountered_overflow: bool,
        heads: CycleHeads,
        last_iteration_provisional_result: Option<X::Result>,
        rebase_entries_kind: Option<RebaseEntriesKind>,
        result: X::Result,
    },
    CycleOnStack {
        entry_node_id: NodeId,
        result: X::Result,
    },
    ProvisionalCacheHit {
        entry_node_id: NodeId,
    },
}

#[derive_where(Debug; X: Cx)]
struct Node<X: Cx> {
    info: GoalInfo<X>,
    parent: Option<NodeId>,
    kind: NodeKind<X>,
}

#[derive_where(Debug; X: Cx)]
pub(super) struct Cycle<X: Cx> {
    pub node_id: NodeId,
    pub provisional_results: HashMap<StackDepth, X::Result>,
}

#[derive_where(Debug, Default; X: Cx)]
pub(super) struct SearchTree<X: Cx> {
    nodes: IndexVec<NodeId, Node<X>>,
    cycles: IndexVec<CycleId, Cycle<X>>,
}

impl<X: Cx> SearchTree<X> {
    pub(super) fn create_node(
        &mut self,
        stack: &Stack<X>,
        input: X::Input,
        step_kind_from_parent: PathKind,
        available_depth: AvailableDepth,
    ) -> NodeId {
        let info = GoalInfo { input, step_kind_from_parent, available_depth };
        let parent = stack.last().map(|e| e.node_id);
        self.nodes.push(Node {
            info,
            parent,
            kind: NodeKind::InProgress {
                cycles_start: self.cycles.next_index(),
                last_iteration_provisional_result: None,
                rebase_entries_kind: None,
            },
        })
    }

    pub(super) fn global_cache_hit(&mut self, node_id: NodeId) {
        debug_assert_eq!(node_id, self.nodes.last_index().unwrap());
        debug_assert!(matches!(self.nodes[node_id].kind, NodeKind::InProgress { .. }));
        self.nodes.pop();
    }

    pub(super) fn provisional_cache_hit(
        &mut self,
        node_id: NodeId,
        entry_node_id: NodeId,
        provisional_results: HashMap<StackDepth, X::Result>,
    ) {
        debug_assert_eq!(node_id, self.nodes.last_index().unwrap());
        debug_assert!(matches!(self.nodes[node_id].kind, NodeKind::InProgress { .. }));
        self.cycles.push(Cycle { node_id, provisional_results });
        self.nodes[node_id].kind = NodeKind::ProvisionalCacheHit { entry_node_id };
    }

    pub(super) fn cycle_on_stack(
        &mut self,
        node_id: NodeId,
        entry_node_id: NodeId,
        result: X::Result,
        provisional_results: HashMap<StackDepth, X::Result>,
    ) {
        debug_assert_eq!(node_id, self.nodes.last_index().unwrap());
        debug_assert!(matches!(self.nodes[node_id].kind, NodeKind::InProgress { .. }));
        self.cycles.push(Cycle { node_id, provisional_results });
        self.nodes[node_id].kind = NodeKind::CycleOnStack { entry_node_id, result }
    }

    pub(super) fn finish_evaluation(
        &mut self,
        node_id: NodeId,
        encountered_overflow: bool,
        heads: CycleHeads,
        result: X::Result,
    ) {
        let NodeKind::InProgress {
            cycles_start: _,
            last_iteration_provisional_result,
            rebase_entries_kind,
        } = self.nodes[node_id].kind
        else {
            panic!("unexpected node kind: {:?}", self.nodes[node_id]);
        };
        self.nodes[node_id].kind = NodeKind::Finished {
            encountered_overflow,
            heads,
            result,
            last_iteration_provisional_result,
            rebase_entries_kind,
        }
    }

    pub(super) fn get_cycle(&self, cycle_id: CycleId) -> &Cycle<X> {
        &self.cycles[cycle_id]
    }

    pub(super) fn node_kind_raw(&self, node_id: NodeId) -> &NodeKind<X> {
        &self.nodes[node_id].kind
    }

    pub(super) fn result_matches(&self, prev: NodeId, new: NodeId) -> bool {
        match (&self.nodes[prev].kind, &self.nodes[new].kind) {
            (
                NodeKind::Finished {
                    encountered_overflow: prev_overflow,
                    heads: prev_heads,
                    result: prev_result,
                    rebase_entries_kind: prev_rebase_entries_kind,
                    last_iteration_provisional_result: prev_last_iteration_provisional_result,
                },
                NodeKind::Finished {
                    encountered_overflow: new_overflow,
                    heads: new_heads,
                    result: new_result,
                    rebase_entries_kind: new_rebase_entries_kind,
                    last_iteration_provisional_result: new_last_iteration_provisional_result,
                },
            ) => {
                prev_result == new_result
                    && (*prev_overflow || !*new_overflow)
                    && prev_rebase_entries_kind == new_rebase_entries_kind
                    && prev_last_iteration_provisional_result
                        == new_last_iteration_provisional_result
                    && prev_heads.contains(new_heads)
            }
            (
                NodeKind::CycleOnStack { entry_node_id: _, result: prev },
                NodeKind::CycleOnStack { entry_node_id: _, result: new },
            ) => prev == new,
            (&NodeKind::ProvisionalCacheHit { entry_node_id }, _) => {
                self.result_matches(entry_node_id, new)
            }
            (_, &NodeKind::ProvisionalCacheHit { entry_node_id }) => {
                self.result_matches(prev, entry_node_id)
            }
            result_matches => {
                tracing::debug!(?result_matches);
                false
            }
        }
    }

    pub(super) fn set_rebase_kind(&mut self, node_id: NodeId, rebase_kind: RebaseEntriesKind) {
        if let NodeKind::InProgress {
            cycles_start: _,
            last_iteration_provisional_result: _,
            rebase_entries_kind,
        } = &mut self.nodes[node_id].kind
        {
            let prev = rebase_entries_kind.replace(rebase_kind);
            debug_assert!(prev.is_none());
        } else {
            panic!("unexpected node kind: {:?}", self.nodes[node_id]);
        }
    }

    pub(super) fn rerun_get_and_reset_cycles(
        &mut self,
        node_id: NodeId,
        provisional_result: X::Result,
    ) -> Range<CycleId> {
        if let NodeKind::InProgress {
            cycles_start,
            last_iteration_provisional_result,
            rebase_entries_kind,
        } = &mut self.nodes[node_id].kind
        {
            debug_assert!(rebase_entries_kind.is_none());
            let prev = *cycles_start;
            *cycles_start = self.cycles.next_index();
            *last_iteration_provisional_result = Some(provisional_result);
            prev..self.cycles.next_index()
        } else {
            panic!("unexpected node kind: {:?}", self.nodes[node_id]);
        }
    }

    pub(super) fn get_heads(&self, node_id: NodeId) -> &CycleHeads {
        if let NodeKind::Finished { heads, .. } = &self.nodes[node_id].kind {
            heads
        } else {
            panic!("unexpected node kind: {:?}", self.nodes[node_id]);
        }
    }

    pub(super) fn goal_or_parent_was_reevaluated(
        &self,
        cycle_head: NodeId,
        was_reevaluated: &HashSet<NodeId>,
        mut node_id: NodeId,
    ) -> bool {
        loop {
            if node_id == cycle_head {
                return false;
            } else if was_reevaluated.contains(&node_id) {
                return true;
            } else {
                node_id = self.nodes[node_id].parent.unwrap();
            }
        }
    }

    /// Compute the list of parents of `node_id` until encountering the node
    /// `until`. We're excluding `until` and are including `node_id`.
    pub(super) fn compute_rev_stack(
        &self,
        mut node_id: NodeId,
        until: NodeId,
    ) -> Vec<(NodeId, GoalInfo<X>)> {
        let mut rev_stack = Vec::new();
        loop {
            if node_id == until {
                return rev_stack;
            }

            let node = &self.nodes[node_id];
            rev_stack.push((node_id, node.info));
            node_id = node.parent.unwrap();
        }
    }
}
