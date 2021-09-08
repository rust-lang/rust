use rustc_data_structures::graph::scc::Sccs;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, VarianceDiagInfo};
use std::fmt;
use std::ops::Index;

use crate::type_check::Locations;

crate mod graph;

/// A set of NLL region constraints. These include "outlives"
/// constraints of the form `R1: R2`. Each constraint is identified by
/// a unique `OutlivesConstraintIndex` and you can index into the set
/// (`constraint_set[i]`) to access the constraint details.
#[derive(Clone, Default)]
crate struct OutlivesConstraintSet<'tcx> {
    outlives: IndexVec<OutlivesConstraintIndex, OutlivesConstraint<'tcx>>,
}

impl<'tcx> OutlivesConstraintSet<'tcx> {
    crate fn push(&mut self, constraint: OutlivesConstraint<'tcx>) {
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
    crate fn graph(&self, num_region_vars: usize) -> graph::NormalConstraintGraph {
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
    crate fn compute_sccs(
        &self,
        constraint_graph: &graph::NormalConstraintGraph,
        static_region: RegionVid,
    ) -> Sccs<RegionVid, ConstraintSccIndex> {
        let region_graph = &constraint_graph.region_graph(self, static_region);
        Sccs::new(region_graph)
    }

    crate fn outlives(&self) -> &IndexVec<OutlivesConstraintIndex, OutlivesConstraint<'tcx>> {
        &self.outlives
    }
}

impl<'tcx> Index<OutlivesConstraintIndex> for OutlivesConstraintSet<'tcx> {
    type Output = OutlivesConstraint<'tcx>;

    fn index(&self, i: OutlivesConstraintIndex) -> &Self::Output {
        &self.outlives[i]
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OutlivesConstraint<'tcx> {
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

    /// Variance diagnostic information
    pub variance_info: VarianceDiagInfo<'tcx>,
}

impl<'tcx> fmt::Debug for OutlivesConstraint<'tcx> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "({:?}: {:?}) due to {:?} ({:?})",
            self.sup, self.sub, self.locations, self.variance_info
        )
    }
}

rustc_index::newtype_index! {
    pub struct OutlivesConstraintIndex {
        DEBUG_FORMAT = "OutlivesConstraintIndex({})"
    }
}

rustc_index::newtype_index! {
    pub struct ConstraintSccIndex {
        DEBUG_FORMAT = "ConstraintSccIndex({})"
    }
}
