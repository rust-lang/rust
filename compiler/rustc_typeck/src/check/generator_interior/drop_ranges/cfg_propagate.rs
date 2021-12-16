use super::{DropRangesBuilder, PostOrderId};
use rustc_index::{bit_set::BitSet, vec::IndexVec};
use std::collections::BTreeMap;

impl DropRangesBuilder {
    pub fn propagate_to_fixpoint(&mut self) {
        trace!("before fixpoint: {:#?}", self);
        let preds = self.compute_predecessors();

        trace!("predecessors: {:#?}", preds.iter_enumerated().collect::<BTreeMap<_, _>>());

        let mut new_state = BitSet::new_empty(self.num_values());

        let mut propagate = || {
            let mut changed = false;
            for id in self.nodes.indices() {
                if id.index() == 0 {
                    new_state.clear();
                } else {
                    // If we are not the start node and we have no predecessors, treat
                    // everything as dropped because there's no way to get here anyway.
                    new_state.insert_all();
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

                changed |= self.nodes[id].drop_state.intersect(&new_state);
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
            // If the node has no explicit successors, we assume that control
            // will from this node into the next one.
            //
            // If there are successors listed, then we assume that all
            // possible successors are given and we do not include the default.
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
}
