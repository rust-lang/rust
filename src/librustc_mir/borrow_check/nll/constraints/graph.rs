use crate::borrow_check::nll::type_check::Locations;
use crate::borrow_check::nll::constraints::OutlivesConstraintIndex;
use crate::borrow_check::nll::constraints::{OutlivesConstraintSet, OutlivesConstraint};
use rustc::mir::ConstraintCategory;
use rustc::ty::RegionVid;
use rustc_data_structures::graph;
use rustc_data_structures::indexed_vec::IndexVec;
use syntax_pos::DUMMY_SP;

/// The construct graph organizes the constraints by their end-points.
/// It can be used to view a `R1: R2` constraint as either an edge `R1
/// -> R2` or `R2 -> R1` depending on the direction type `D`.
crate struct ConstraintGraph<D: ConstraintGraphDirecton> {
    _direction: D,
    first_constraints: IndexVec<RegionVid, Option<OutlivesConstraintIndex>>,
    next_constraints: IndexVec<OutlivesConstraintIndex, Option<OutlivesConstraintIndex>>,
}

crate type NormalConstraintGraph = ConstraintGraph<Normal>;

crate type ReverseConstraintGraph = ConstraintGraph<Reverse>;

/// Marker trait that controls whether a `R1: R2` constraint
/// represents an edge `R1 -> R2` or `R2 -> R1`.
crate trait ConstraintGraphDirecton: Copy + 'static {
    fn start_region(c: &OutlivesConstraint) -> RegionVid;
    fn end_region(c: &OutlivesConstraint) -> RegionVid;
    fn is_normal() -> bool;
}

/// In normal mode, a `R1: R2` constraint results in an edge `R1 ->
/// R2`. This is what we use when constructing the SCCs for
/// inference. This is because we compute the value of R1 by union'ing
/// all the things that it relies on.
#[derive(Copy, Clone, Debug)]
crate struct Normal;

impl ConstraintGraphDirecton for Normal {
    fn start_region(c: &OutlivesConstraint) -> RegionVid {
        c.sup
    }

    fn end_region(c: &OutlivesConstraint) -> RegionVid {
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
crate struct Reverse;

impl ConstraintGraphDirecton for Reverse {
    fn start_region(c: &OutlivesConstraint) -> RegionVid {
        c.sub
    }

    fn end_region(c: &OutlivesConstraint) -> RegionVid {
        c.sup
    }

    fn is_normal() -> bool {
        false
    }
}

impl<D: ConstraintGraphDirecton> ConstraintGraph<D> {
    /// Creates a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    crate fn new(
        direction: D,
        set: &OutlivesConstraintSet,
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

        Self {
            _direction: direction,
            first_constraints,
            next_constraints,
        }
    }

    /// Given the constraint set from which this graph was built
    /// creates a region graph so that you can iterate over *regions*
    /// and not constraints.
    crate fn region_graph<'rg>(
        &'rg self,
        set: &'rg OutlivesConstraintSet,
        static_region: RegionVid,
    ) -> RegionGraph<'rg, D> {
        RegionGraph::new(set, self, static_region)
    }

    /// Given a region `R`, iterate over all constraints `R: R1`.
    crate fn outgoing_edges<'a>(
        &'a self,
        region_sup: RegionVid,
        constraints: &'a OutlivesConstraintSet,
        static_region: RegionVid,
    ) -> Edges<'a, D> {
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
            Edges {
                graph: self,
                constraints,
                pointer: first,
                next_static_idx: None,
                static_region,
           }
        }
    }
}

crate struct Edges<'s, D: ConstraintGraphDirecton> {
    graph: &'s ConstraintGraph<D>,
    constraints: &'s OutlivesConstraintSet,
    pointer: Option<OutlivesConstraintIndex>,
    next_static_idx: Option<usize>,
    static_region: RegionVid,
}

impl<'s, D: ConstraintGraphDirecton> Iterator for Edges<'s, D> {
    type Item = OutlivesConstraint;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.pointer {
            self.pointer = self.graph.next_constraints[p];

            Some(self.constraints[p])
        } else if let Some(next_static_idx) = self.next_static_idx {
            self.next_static_idx =
                if next_static_idx == (self.graph.first_constraints.len() - 1) {
                    None
                } else {
                    Some(next_static_idx + 1)
                };

            Some(OutlivesConstraint {
                sup: self.static_region,
                sub: next_static_idx.into(),
                locations: Locations::All(DUMMY_SP),
                category: ConstraintCategory::Internal,
            })
        } else {
            None
        }
    }
}

/// This struct brings together a constraint set and a (normal, not
/// reverse) constraint graph. It implements the graph traits and is
/// usd for doing the SCC computation.
crate struct RegionGraph<'s, D: ConstraintGraphDirecton> {
    set: &'s OutlivesConstraintSet,
    constraint_graph: &'s ConstraintGraph<D>,
    static_region: RegionVid,
}

impl<'s, D: ConstraintGraphDirecton> RegionGraph<'s, D> {
    /// Creates a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    crate fn new(
        set: &'s OutlivesConstraintSet,
        constraint_graph: &'s ConstraintGraph<D>,
        static_region: RegionVid,
    ) -> Self {
        Self {
            set,
            constraint_graph,
            static_region,
        }
    }

    /// Given a region `R`, iterate over all regions `R1` such that
    /// there exists a constraint `R: R1`.
    crate fn outgoing_regions(&self, region_sup: RegionVid) -> Successors<'_, D> {
        Successors {
            edges: self.constraint_graph.outgoing_edges(region_sup, self.set, self.static_region),
        }
    }
}

crate struct Successors<'s, D: ConstraintGraphDirecton> {
    edges: Edges<'s, D>,
}

impl<'s, D: ConstraintGraphDirecton> Iterator for Successors<'s, D> {
    type Item = RegionVid;

    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|c| D::end_region(&c))
    }
}

impl<'s, D: ConstraintGraphDirecton> graph::DirectedGraph for RegionGraph<'s, D> {
    type Node = RegionVid;
}

impl<'s, D: ConstraintGraphDirecton> graph::WithNumNodes for RegionGraph<'s, D> {
    fn num_nodes(&self) -> usize {
        self.constraint_graph.first_constraints.len()
    }
}

impl<'s, D: ConstraintGraphDirecton> graph::WithSuccessors for RegionGraph<'s, D> {
    fn successors(
        &self,
        node: Self::Node,
    ) -> <Self as graph::GraphSuccessors<'_>>::Iter {
        self.outgoing_regions(node)
    }
}

impl<'s, 'graph, D: ConstraintGraphDirecton> graph::GraphSuccessors<'graph> for RegionGraph<'s, D> {
    type Item = RegionVid;
    type Iter = Successors<'graph, D>;
}
