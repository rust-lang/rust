use std::collections::BTreeMap;
use std::fmt::Debug;
use std::mem::swap;

use rustc_hir::{HirId, HirIdMap};
use rustc_index::bit_set::BitSet;
use rustc_index::vec::IndexVec;
use rustc_middle::hir::map::Map;

use super::for_each_consumable;

rustc_index::newtype_index! {
    pub struct PostOrderId {
        DEBUG_FORMAT = "id({})",
    }
}

rustc_index::newtype_index! {
    pub struct HirIdIndex {
        DEBUG_FORMAT = "hidx({})",
    }
}

pub struct DropRanges {
    hir_id_map: HirIdMap<HirIdIndex>,
    nodes: IndexVec<PostOrderId, NodeInfo>,
    deferred_edges: Vec<(usize, HirId)>,
    // FIXME: This should only be used for loops and break/continue.
    post_order_map: HirIdMap<usize>,
}

impl Debug for DropRanges {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropRanges")
            .field("hir_id_map", &self.hir_id_map)
            .field("post_order_maps", &self.post_order_map)
            .field("nodes", &self.nodes.iter_enumerated().collect::<BTreeMap<_, _>>())
            .finish()
    }
}

/// DropRanges keeps track of what values are definitely dropped at each point in the code.
///
/// Values of interest are defined by the hir_id of their place. Locations in code are identified
/// by their index in the post-order traversal. At its core, DropRanges maps
/// (hir_id, post_order_id) -> bool, where a true value indicates that the value is definitely
/// dropped at the point of the node identified by post_order_id.
impl DropRanges {
    pub fn new(hir_ids: impl Iterator<Item = HirId>, hir: &Map<'_>, num_exprs: usize) -> Self {
        let mut hir_id_map = HirIdMap::<HirIdIndex>::default();
        let mut next = <_>::from(0u32);
        for hir_id in hir_ids {
            for_each_consumable(hir_id, hir.find(hir_id), |hir_id| {
                if !hir_id_map.contains_key(&hir_id) {
                    hir_id_map.insert(hir_id, next);
                    next = <_>::from(next.index() + 1);
                }
            });
        }
        debug!("hir_id_map: {:?}", hir_id_map);
        let num_values = hir_id_map.len();
        Self {
            hir_id_map,
            nodes: IndexVec::from_fn_n(|_| NodeInfo::new(num_values), num_exprs + 1),
            deferred_edges: <_>::default(),
            post_order_map: <_>::default(),
        }
    }

    fn hidx(&self, hir_id: HirId) -> HirIdIndex {
        *self.hir_id_map.get(&hir_id).unwrap()
    }

    pub fn is_dropped_at(&mut self, hir_id: HirId, location: usize) -> bool {
        self.hir_id_map
            .get(&hir_id)
            .copied()
            .map_or(false, |hir_id| self.expect_node(location.into()).drop_state.contains(hir_id))
    }

    /// Returns the number of values (hir_ids) that are tracked
    fn num_values(&self) -> usize {
        self.hir_id_map.len()
    }

    /// Adds an entry in the mapping from HirIds to PostOrderIds
    ///
    /// Needed so that `add_control_edge_hir_id` can work.
    pub fn add_node_mapping(&mut self, hir_id: HirId, post_order_id: usize) {
        self.post_order_map.insert(hir_id, post_order_id);
    }

    /// Returns a reference to the NodeInfo for a node, panicking if it does not exist
    fn expect_node(&self, id: PostOrderId) -> &NodeInfo {
        &self.nodes[id]
    }

    fn node_mut(&mut self, id: PostOrderId) -> &mut NodeInfo {
        let size = self.num_values();
        self.nodes.ensure_contains_elem(id, || NodeInfo::new(size));
        &mut self.nodes[id]
    }

    pub fn add_control_edge(&mut self, from: usize, to: usize) {
        trace!("adding control edge from {} to {}", from, to);
        self.node_mut(from.into()).successors.push(to.into());
    }

    /// Like add_control_edge, but uses a hir_id as the target.
    ///
    /// This can be used for branches where we do not know the PostOrderId of the target yet,
    /// such as when handling `break` or `continue`.
    pub fn add_control_edge_hir_id(&mut self, from: usize, to: HirId) {
        self.deferred_edges.push((from, to));
    }

    /// Looks up PostOrderId for any control edges added by HirId and adds a proper edge for them.
    ///
    /// Should be called after visiting the HIR but before solving the control flow, otherwise some
    /// edges will be missed.
    fn process_deferred_edges(&mut self) {
        let mut edges = vec![];
        swap(&mut edges, &mut self.deferred_edges);
        edges.into_iter().for_each(|(from, to)| {
            let to = *self.post_order_map.get(&to).expect("Expression ID not found");
            trace!("Adding deferred edge from {} to {}", from, to);
            self.add_control_edge(from, to)
        });
    }

    pub fn drop_at(&mut self, value: HirId, location: usize) {
        let value = self.hidx(value);
        self.node_mut(location.into()).drops.push(value);
    }

    pub fn reinit_at(&mut self, value: HirId, location: usize) {
        let value = match self.hir_id_map.get(&value) {
            Some(value) => *value,
            // If there's no value, this is never consumed and therefore is never dropped. We can
            // ignore this.
            None => return,
        };
        self.node_mut(location.into()).reinits.push(value);
    }

    pub fn propagate_to_fixpoint(&mut self) {
        trace!("before fixpoint: {:#?}", self);
        self.process_deferred_edges();
        let preds = self.compute_predecessors();

        trace!("predecessors: {:#?}", preds.iter_enumerated().collect::<BTreeMap<_, _>>());

        let mut propagate = || {
            let mut changed = false;
            for id in self.nodes.indices() {
                let old_state = self.nodes[id].drop_state.clone();
                if preds[id].len() != 0 {
                    self.nodes[id].drop_state = self.nodes[preds[id][0]].drop_state.clone();
                    for pred in &preds[id][1..] {
                        let state = self.nodes[*pred].drop_state.clone();
                        self.nodes[id].drop_state.intersect(&state);
                    }
                } else {
                    self.nodes[id].drop_state = if id.index() == 0 {
                        BitSet::new_empty(self.num_values())
                    } else {
                        // If we are not the start node and we have no predecessors, treat
                        // everything as dropped because there's no way to get here anyway.
                        BitSet::new_filled(self.num_values())
                    };
                };
                for drop in &self.nodes[id].drops.clone() {
                    self.nodes[id].drop_state.insert(*drop);
                }
                for reinit in &self.nodes[id].reinits.clone() {
                    self.nodes[id].drop_state.remove(*reinit);
                }

                changed |= old_state != self.nodes[id].drop_state;
            }

            changed
        };

        while propagate() {}

        trace!("after fixpoint: {:#?}", self);
    }

    fn compute_predecessors(&self) -> IndexVec<PostOrderId, Vec<PostOrderId>> {
        let mut preds = IndexVec::from_fn_n(|_| vec![], self.nodes.len());
        for (id, node) in self.nodes.iter_enumerated() {
            if node.successors.len() == 0 && id.index() != self.nodes.len() - 1 {
                preds[<_>::from(id.index() + 1)].push(id);
            } else {
                for succ in &node.successors {
                    preds[*succ].push(id);
                }
            }
        }
        preds
    }
}

#[derive(Debug)]
struct NodeInfo {
    /// IDs of nodes that can follow this one in the control flow
    ///
    /// If the vec is empty, then control proceeds to the next node.
    successors: Vec<PostOrderId>,

    /// List of hir_ids that are dropped by this node.
    drops: Vec<HirIdIndex>,

    /// List of hir_ids that are reinitialized by this node.
    reinits: Vec<HirIdIndex>,

    /// Set of values that are definitely dropped at this point.
    drop_state: BitSet<HirIdIndex>,
}

impl NodeInfo {
    fn new(num_values: usize) -> Self {
        Self {
            successors: vec![],
            drops: vec![],
            reinits: vec![],
            drop_state: BitSet::new_filled(num_values),
        }
    }
}
