use rustc_data_structures::graph::scc::Sccs;
use rustc_index::vec::IndexVec;
use rustc_middle::ty::RegionVid;
use std::ops::Index;

crate use rustc_middle::mir::regions::{
    ConstraintSccIndex, OutlivesConstraint, OutlivesConstraintIndex,
};

pub mod graph;

/// A set of NLL region constraints. These include "outlives"
/// constraints of the form `R1: R2`. Each constraint is identified by
/// a unique `OutlivesConstraintIndex` and you can index into the set
/// (`constraint_set[i]`) to access the constraint details.
#[derive(Clone, Default)]
pub struct OutlivesConstraintSet {
    outlives: IndexVec<OutlivesConstraintIndex, OutlivesConstraint>,
}

impl OutlivesConstraintSet {
    pub fn push(&mut self, constraint: OutlivesConstraint) {
        debug!(
            "OutlivesConstraintSet::push({:?}: {:?} @ {:?}",
            constraint.sup, constraint.sub, constraint.locations
        );
        if constraint.sup == constraint.sub {
            // 'a: 'a is pretty uninteresting
            return;
        }
        self.outlives.push(constraint);
    }

    /// Constructs a "normal" graph from the constraint set; the graph makes it
    /// easy to find the constraints affecting a particular region.
    ///
    /// N.B., this graph contains a "frozen" view of the current
    /// constraints. Any new constraints added to the `OutlivesConstraintSet`
    /// after the graph is built will not be present in the graph.
    pub fn graph(&self, num_region_vars: usize) -> graph::NormalConstraintGraph {
        graph::ConstraintGraph::new(graph::Normal, self, num_region_vars)
    }

    /// Like `graph`, but constraints a reverse graph where `R1: R2`
    /// represents an edge `R2 -> R1`.
    crate fn reverse_graph(&self, num_region_vars: usize) -> graph::ReverseConstraintGraph {
        graph::ConstraintGraph::new(graph::Reverse, self, num_region_vars)
    }

    /// Computes cycles (SCCs) in the graph of regions. In particular,
    /// find all regions R1, R2 such that R1: R2 and R2: R1 and group
    /// them into an SCC, and find the relationships between SCCs.
    pub fn compute_sccs(
        &self,
        constraint_graph: &graph::NormalConstraintGraph,
        static_region: RegionVid,
    ) -> Sccs<RegionVid, ConstraintSccIndex> {
        let region_graph = &constraint_graph.region_graph(self, static_region);
        Sccs::new(region_graph)
    }

    crate fn outlives(&self) -> &IndexVec<OutlivesConstraintIndex, OutlivesConstraint> {
        &self.outlives
    }
}

impl Index<OutlivesConstraintIndex> for OutlivesConstraintSet {
    type Output = OutlivesConstraint;

    fn index(&self, i: OutlivesConstraintIndex) -> &Self::Output {
        &self.outlives[i]
    }
}
