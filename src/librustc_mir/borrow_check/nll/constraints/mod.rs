use rustc::mir::ConstraintCategory;
use rustc::ty::RegionVid;
use rustc_data_structures::graph::scc::Sccs;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use borrow_check::nll::type_check::Locations;

use std::fmt;
use std::ops::Deref;

crate mod graph;

#[derive(Clone, Default)]
crate struct ConstraintSet {
    constraints: IndexVec<ConstraintIndex, OutlivesConstraint>,
}

impl ConstraintSet {
    crate fn push(&mut self, constraint: OutlivesConstraint) {
        debug!(
            "ConstraintSet::push({:?}: {:?} @ {:?}",
            constraint.sup, constraint.sub, constraint.locations
        );
        if constraint.sup == constraint.sub {
            // 'a: 'a is pretty uninteresting
            return;
        }
        self.constraints.push(constraint);
    }

    /// Constructs a "normal" graph from the constraint set; the graph makes it
    /// easy to find the constraints affecting a particular region.
    ///
    /// N.B., this graph contains a "frozen" view of the current
    /// constraints.  any new constraints added to the `ConstraintSet`
    /// after the graph is built will not be present in the graph.
    crate fn graph(&self, num_region_vars: usize) -> graph::NormalConstraintGraph {
        graph::ConstraintGraph::new(graph::Normal, self, num_region_vars)
    }

    /// Like `graph`, but constraints a reverse graph where `R1: R2`
    /// represents an edge `R2 -> R1`.
    crate fn reverse_graph(&self, num_region_vars: usize) -> graph::ReverseConstraintGraph {
        graph::ConstraintGraph::new(graph::Reverse, self, num_region_vars)
    }

    /// Compute cycles (SCCs) in the graph of regions. In particular,
    /// find all regions R1, R2 such that R1: R2 and R2: R1 and group
    /// them into an SCC, and find the relationships between SCCs.
    crate fn compute_sccs(
        &self,
        constraint_graph: &graph::NormalConstraintGraph,
        static_region: RegionVid,
    ) -> Sccs<RegionVid, ConstraintSccIndex> {
        let region_graph = &constraint_graph.region_graph(self, static_region);
        Sccs::new(region_graph)
    }
}

impl Deref for ConstraintSet {
    type Target = IndexVec<ConstraintIndex, OutlivesConstraint>;

    fn deref(&self) -> &Self::Target {
        &self.constraints
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutlivesConstraint {
    // NB. The ordering here is not significant for correctness, but
    // it is for convenience. Before we dump the constraints in the
    // debugging logs, we sort them, and we'd like the "super region"
    // to be first, etc. (In particular, span should remain last.)
    /// The region SUP must outlive SUB...
    pub sup: RegionVid,

    /// Region that must be outlived.
    pub sub: RegionVid,

    /// Where did this constraint arise?
    pub locations: Locations,

    /// What caused this constraint?
    pub category: ConstraintCategory,
}

impl fmt::Debug for OutlivesConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "({:?}: {:?}) due to {:?}",
            self.sup, self.sub, self.locations
        )
    }
}

newtype_index! {
    pub struct ConstraintIndex {
        DEBUG_FORMAT = "ConstraintIndex({})"
    }
}

newtype_index! {
    pub struct ConstraintSccIndex {
        DEBUG_FORMAT = "ConstraintSccIndex({})"
    }
}
