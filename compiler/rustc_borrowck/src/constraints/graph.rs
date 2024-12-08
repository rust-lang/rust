use rustc_data_structures::graph;
use rustc_index::IndexVec;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, VarianceDiagInfo};
use rustc_span::DUMMY_SP;

use crate::constraints::{OutlivesConstraint, OutlivesConstraintIndex, OutlivesConstraintSet};
use crate::type_check::Locations;

/// The construct graph organizes the constraints by their end-points.
/// It can be used to view a `R1: R2` constraint as either an edge `R1
/// -> R2` or `R2 -> R1` depending on the direction type `D`.
pub(crate) struct ConstraintGraph<D: ConstraintGraphDirection> {
    _direction: D,
    first_constraints: IndexVec<RegionVid, Option<OutlivesConstraintIndex>>,
    next_constraints: IndexVec<OutlivesConstraintIndex, Option<OutlivesConstraintIndex>>,
}

pub(crate) type NormalConstraintGraph = ConstraintGraph<Normal>;

pub(crate) type ReverseConstraintGraph = ConstraintGraph<Reverse>;

/// Marker trait that controls whether a `R1: R2` constraint
/// represents an edge `R1 -> R2` or `R2 -> R1`.
pub(crate) trait ConstraintGraphDirection: Copy + 'static {
    fn start_region(c: &OutlivesConstraint<'_>) -> RegionVid;
    fn end_region(c: &OutlivesConstraint<'_>) -> RegionVid;
    fn is_normal() -> bool;
}

/// In normal mode, a `R1: R2` constraint results in an edge `R1 ->
/// R2`. This is what we use when constructing the SCCs for
/// inference. This is because we compute the value of R1 by union'ing
/// all the things that it relies on.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Normal;

impl ConstraintGraphDirection for Normal {
    fn start_region(c: &OutlivesConstraint<'_>) -> RegionVid {
        c.sup
    }

    fn end_region(c: &OutlivesConstraint<'_>) -> RegionVid {
        c.sub
    }

    fn is_normal() -> bool {
        true
    }
}

/// In reverse mode, a `R1: R2` constraint results in an edge `R2 ->
/// R1`. We use this for optimizing liveness computation, because then
/// we wish to iterate from a region (e.g., R2) to all the regions
/// that will outlive it (e.g., R1).
#[derive(Copy, Clone, Debug)]
pub(crate) struct Reverse;

impl ConstraintGraphDirection for Reverse {
    fn start_region(c: &OutlivesConstraint<'_>) -> RegionVid {
        c.sub
    }

    fn end_region(c: &OutlivesConstraint<'_>) -> RegionVid {
        c.sup
    }

    fn is_normal() -> bool {
        false
    }
}

impl<D: ConstraintGraphDirection> ConstraintGraph<D> {
    /// Creates a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    pub(crate) fn new(
        direction: D,
        set: &OutlivesConstraintSet<'_>,
        num_region_vars: usize,
    ) -> Self {
        let mut first_constraints = IndexVec::from_elem_n(None, num_region_vars);
        let mut next_constraints = IndexVec::from_elem(None, &set.outlives);

        for (idx, constraint) in set.outlives.iter_enumerated().rev() {
            let head = &mut first_constraints[D::start_region(constraint)];
            let next = &mut next_constraints[idx];
            debug_assert!(next.is_none());
            *next = *head;
            *head = Some(idx);
        }

        Self { _direction: direction, first_constraints, next_constraints }
    }

    /// Given the constraint set from which this graph was built
    /// creates a region graph so that you can iterate over *regions*
    /// and not constraints.
    pub(crate) fn region_graph<'a, 'tcx>(
        &'a self,
        set: &'a OutlivesConstraintSet<'tcx>,
        static_region: RegionVid,
    ) -> RegionGraph<'a, 'tcx, D> {
        RegionGraph::new(set, self, static_region)
    }

    /// Given a region `R`, iterate over all constraints `R: R1`.
    pub(crate) fn outgoing_edges<'a, 'tcx>(
        &'a self,
        region_sup: RegionVid,
        constraints: &'a OutlivesConstraintSet<'tcx>,
        static_region: RegionVid,
    ) -> Edges<'a, 'tcx, D> {
        //if this is the `'static` region and the graph's direction is normal,
        //then setup the Edges iterator to return all regions #53178
        if region_sup == static_region && D::is_normal() {
            Edges {
                graph: self,
                constraints,
                pointer: None,
                next_static_idx: Some(0),
                static_region,
            }
        } else {
            //otherwise, just setup the iterator as normal
            let first = self.first_constraints[region_sup];
            Edges { graph: self, constraints, pointer: first, next_static_idx: None, static_region }
        }
    }
}

pub(crate) struct Edges<'a, 'tcx, D: ConstraintGraphDirection> {
    graph: &'a ConstraintGraph<D>,
    constraints: &'a OutlivesConstraintSet<'tcx>,
    pointer: Option<OutlivesConstraintIndex>,
    next_static_idx: Option<usize>,
    static_region: RegionVid,
}

impl<'a, 'tcx, D: ConstraintGraphDirection> Iterator for Edges<'a, 'tcx, D> {
    type Item = OutlivesConstraint<'tcx>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.pointer {
            self.pointer = self.graph.next_constraints[p];

            Some(self.constraints[p])
        } else if let Some(next_static_idx) = self.next_static_idx {
            self.next_static_idx = if next_static_idx == (self.graph.first_constraints.len() - 1) {
                None
            } else {
                Some(next_static_idx + 1)
            };

            Some(OutlivesConstraint {
                sup: self.static_region,
                sub: next_static_idx.into(),
                locations: Locations::All(DUMMY_SP),
                span: DUMMY_SP,
                category: ConstraintCategory::Internal,
                variance_info: VarianceDiagInfo::default(),
                from_closure: false,
            })
        } else {
            None
        }
    }
}

/// This struct brings together a constraint set and a (normal, not
/// reverse) constraint graph. It implements the graph traits and is
/// usd for doing the SCC computation.
pub(crate) struct RegionGraph<'a, 'tcx, D: ConstraintGraphDirection> {
    set: &'a OutlivesConstraintSet<'tcx>,
    constraint_graph: &'a ConstraintGraph<D>,
    static_region: RegionVid,
}

impl<'a, 'tcx, D: ConstraintGraphDirection> RegionGraph<'a, 'tcx, D> {
    /// Creates a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    pub(crate) fn new(
        set: &'a OutlivesConstraintSet<'tcx>,
        constraint_graph: &'a ConstraintGraph<D>,
        static_region: RegionVid,
    ) -> Self {
        Self { set, constraint_graph, static_region }
    }

    /// Given a region `R`, iterate over all regions `R1` such that
    /// there exists a constraint `R: R1`.
    pub(crate) fn outgoing_regions(&self, region_sup: RegionVid) -> Successors<'a, 'tcx, D> {
        Successors {
            edges: self.constraint_graph.outgoing_edges(region_sup, self.set, self.static_region),
        }
    }
}

pub(crate) struct Successors<'a, 'tcx, D: ConstraintGraphDirection> {
    edges: Edges<'a, 'tcx, D>,
}

impl<'a, 'tcx, D: ConstraintGraphDirection> Iterator for Successors<'a, 'tcx, D> {
    type Item = RegionVid;

    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|c| D::end_region(&c))
    }
}

impl<'a, 'tcx, D: ConstraintGraphDirection> graph::DirectedGraph for RegionGraph<'a, 'tcx, D> {
    type Node = RegionVid;

    fn num_nodes(&self) -> usize {
        self.constraint_graph.first_constraints.len()
    }
}

impl<'a, 'tcx, D: ConstraintGraphDirection> graph::Successors for RegionGraph<'a, 'tcx, D> {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.outgoing_regions(node)
    }
}
