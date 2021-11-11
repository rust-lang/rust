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
}

/// DropRanges keeps track of what values are definitely dropped at each point in the code.
///
/// Values of interest are defined by the hir_id of their place. Locations in code are identified
/// by their index in the post-order traversal. At its core, DropRanges maps
/// (hir_id, post_order_id) -> bool, where a true value indicates that the value is definitely
/// dropped at the point of the node identified by post_order_id.
impl DropRanges {
    pub fn new(hir_ids: impl Iterator<Item = HirId>, hir: &Map<'_>) -> Self {
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
        Self { hir_id_map, nodes: <_>::default() }
    }

    fn hidx(&self, hir_id: HirId) -> HirIdIndex {
        *self.hir_id_map.get(&hir_id).unwrap()
    }

    pub fn is_dropped_at(&mut self, hir_id: HirId, location: usize) -> bool {
        self.hir_id_map
            .get(&hir_id)
            .copied()
            .map_or(false, |hir_id| self.node(location.into()).drop_state.contains(hir_id))
    }

    /// Returns the number of values (hir_ids) that are tracked
    fn num_values(&self) -> usize {
        self.hir_id_map.len()
    }

    fn node(&mut self, id: PostOrderId) -> &NodeInfo {
        let size = self.num_values();
        self.nodes.ensure_contains_elem(id, || NodeInfo::new(size));
        &self.nodes[id]
    }

    fn node_mut(&mut self, id: PostOrderId) -> &mut NodeInfo {
        let size = self.num_values();
        self.nodes.ensure_contains_elem(id, || NodeInfo::new(size));
        &mut self.nodes[id]
    }

    pub fn add_control_edge(&mut self, from: usize, to: usize) {
        self.node_mut(from.into()).successors.push(to.into());
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
        while self.propagate() {}
    }

    fn propagate(&mut self) -> bool {
        let mut visited = BitSet::new_empty(self.nodes.len());

        self.visit(&mut visited, PostOrderId::from(0usize), PostOrderId::from(0usize), false)
    }

    fn visit(
        &mut self,
        visited: &mut BitSet<PostOrderId>,
        id: PostOrderId,
        pred_id: PostOrderId,
        mut changed: bool,
    ) -> bool {
        if visited.contains(id) {
            return changed;
        }
        visited.insert(id);

        changed &= self.nodes[id].merge_with(&self.nodes[pred_id]);

        if self.nodes[id].successors.len() == 0 {
            self.visit(visited, PostOrderId::from(id.index() + 1), id, changed)
        } else {
            self.nodes[id]
                .successors
                .iter()
                .fold(changed, |changed, &succ| self.visit(visited, succ, id, changed))
        }
    }
}

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
            drop_state: BitSet::new_empty(num_values),
        }
    }

    fn merge_with(&mut self, other: &NodeInfo) -> bool {
        let mut changed = false;
        for place in &self.drops {
            if !self.drop_state.contains(place) && !self.reinits.contains(&place) {
                changed = true;
                self.drop_state.insert(place);
            }
        }

        for place in &self.reinits {
            if self.drop_state.contains(place) {
                changed = true;
                self.drop_state.remove(place);
            }
        }

        changed
    }
}
