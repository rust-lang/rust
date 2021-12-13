use super::{DropRanges, PostOrderId};
use rustc_index::{bit_set::BitSet, vec::IndexVec};
use std::collections::BTreeMap;
use std::mem::swap;

impl DropRanges {
    pub fn propagate_to_fixpoint(&mut self) {
        trace!("before fixpoint: {:#?}", self);
        self.process_deferred_edges();
        let preds = self.compute_predecessors();

        trace!("predecessors: {:#?}", preds.iter_enumerated().collect::<BTreeMap<_, _>>());

        let mut propagate = || {
            let mut changed = false;
            for id in self.nodes.indices() {
                let old_state = self.nodes[id].drop_state.clone();
                let mut new_state = if id.index() == 0 {
                    BitSet::new_empty(self.num_values())
                } else {
                    // If we are not the start node and we have no predecessors, treat
                    // everything as dropped because there's no way to get here anyway.
                    BitSet::new_filled(self.num_values())
                };

                for pred in &preds[id] {
                    let state = &self.nodes[*pred].drop_state;
                    new_state.intersect(state);
                }

                for drop in &self.nodes[id].drops {
                    new_state.insert(*drop);
                }

                for reinit in &self.nodes[id].reinits {
                    new_state.remove(*reinit);
                }

                changed |= old_state != new_state;
                self.nodes[id].drop_state = new_state;
            }

            changed
        };

        while propagate() {
            trace!("drop_state changed, re-running propagation");
        }

        trace!("after fixpoint: {:#?}", self);
    }

    fn compute_predecessors(&self) -> IndexVec<PostOrderId, Vec<PostOrderId>> {
        let mut preds = IndexVec::from_fn_n(|_| vec![], self.nodes.len());
        for (id, node) in self.nodes.iter_enumerated() {
            if node.successors.len() == 0 && id.index() != self.nodes.len() - 1 {
                preds[id + 1].push(id);
            } else {
                for succ in &node.successors {
                    preds[*succ].push(id);
                }
            }
        }
        preds
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
}
