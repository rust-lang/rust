// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::constraints::{ConstraintIndex, ConstraintSet, OutlivesConstraint};
use rustc::ty::RegionVid;
use rustc_data_structures::graph;
use rustc_data_structures::indexed_vec::IndexVec;

/// The construct graph organizes the constraints by their end-points.
/// It can be used to view a `R1: R2` constraint as either an edge `R1
/// -> R2` or `R2 -> R1` depending on the direction type `D`.
crate struct ConstraintGraph<D: ConstraintGraphDirecton> {
    _direction: D,
    first_constraints: IndexVec<RegionVid, Option<ConstraintIndex>>,
    next_constraints: IndexVec<ConstraintIndex, Option<ConstraintIndex>>,
}

crate type NormalConstraintGraph = ConstraintGraph<Normal>;

crate type ReverseConstraintGraph = ConstraintGraph<Reverse>;

/// Marker trait that controls whether a `R1: R2` constraint
/// represents an edge `R1 -> R2` or `R2 -> R1`.
crate trait ConstraintGraphDirecton: Copy + 'static {
    fn start_region(c: &OutlivesConstraint) -> RegionVid;
    fn end_region(c: &OutlivesConstraint) -> RegionVid;
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
}

impl<D: ConstraintGraphDirecton> ConstraintGraph<D> {
    /// Create a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    crate fn new(
        direction: D,
        set: &ConstraintSet,
        num_region_vars: usize,
    ) -> Self {
        let mut first_constraints = IndexVec::from_elem_n(None, num_region_vars);
        let mut next_constraints = IndexVec::from_elem(None, &set.constraints);

        for (idx, constraint) in set.constraints.iter_enumerated().rev() {
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
    crate fn region_graph<'rg>(&'rg self, set: &'rg ConstraintSet) -> RegionGraph<'rg, D> {
        RegionGraph::new(set, self)
    }

    /// Given a region `R`, iterate over all constraints `R: R1`.
    crate fn outgoing_edges<'a>(
        &'a self,
        region_sup: RegionVid,
        constraints: &'a ConstraintSet,
    ) -> Edges<'a, D> {
        let first = self.first_constraints[region_sup];
        Edges {
            graph: self,
            constraints,
            pointer: first,
        }
    }
}

crate struct Edges<'s, D: ConstraintGraphDirecton> {
    graph: &'s ConstraintGraph<D>,
    constraints: &'s ConstraintSet,
    pointer: Option<ConstraintIndex>,
}

impl<'s, D: ConstraintGraphDirecton> Iterator for Edges<'s, D> {
    type Item = OutlivesConstraint;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.pointer {
            self.pointer = self.graph.next_constraints[p];
            Some(self.constraints[p])
        } else {
            None
        }
    }
}

/// This struct brings together a constraint set and a (normal, not
/// reverse) constraint graph. It implements the graph traits and is
/// usd for doing the SCC computation.
crate struct RegionGraph<'s, D: ConstraintGraphDirecton> {
    set: &'s ConstraintSet,
    constraint_graph: &'s ConstraintGraph<D>,
}

impl<'s, D: ConstraintGraphDirecton> RegionGraph<'s, D> {
    /// Create a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    crate fn new(set: &'s ConstraintSet, constraint_graph: &'s ConstraintGraph<D>) -> Self {
        Self {
            set,
            constraint_graph,
        }
    }

    /// Given a region `R`, iterate over all regions `R1` such that
    /// there exists a constraint `R: R1`.
    crate fn outgoing_regions(&self, region_sup: RegionVid) -> Successors<'_, D> {
        Successors {
            edges: self.constraint_graph.outgoing_edges(region_sup, self.set),
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
    fn successors<'graph>(
        &'graph self,
        node: Self::Node,
    ) -> <Self as graph::GraphSuccessors<'graph>>::Iter {
        self.outgoing_regions(node)
    }
}

impl<'s, 'graph, D: ConstraintGraphDirecton> graph::GraphSuccessors<'graph> for RegionGraph<'s, D> {
    type Item = RegionVid;
    type Iter = Successors<'graph, D>;
}
