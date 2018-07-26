// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::constraints::{ConstraintIndex, ConstraintSet};
use rustc::ty::RegionVid;
use rustc_data_structures::graph;
use rustc_data_structures::indexed_vec::IndexVec;

crate struct ConstraintGraph {
    first_constraints: IndexVec<RegionVid, Option<ConstraintIndex>>,
    next_constraints: IndexVec<ConstraintIndex, Option<ConstraintIndex>>,
}

impl ConstraintGraph {
    /// Create a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    crate fn new(set: &ConstraintSet, num_region_vars: usize) -> Self {
        let mut first_constraints = IndexVec::from_elem_n(None, num_region_vars);
        let mut next_constraints = IndexVec::from_elem(None, &set.constraints);

        for (idx, constraint) in set.constraints.iter_enumerated().rev() {
            let mut head = &mut first_constraints[constraint.sup];
            let mut next = &mut next_constraints[idx];
            debug_assert!(next.is_none());
            *next = *head;
            *head = Some(idx);
        }

        Self {
            first_constraints,
            next_constraints,
        }
    }

    /// Given a region `R`, iterate over all constraints `R: R1`.
    crate fn outgoing_edges(&self, region_sup: RegionVid) -> Edges<'_> {
        let first = self.first_constraints[region_sup];
        Edges {
            graph: self,
            pointer: first,
        }
    }
}

crate struct Edges<'graph> {
    graph: &'graph ConstraintGraph,
    pointer: Option<ConstraintIndex>,
}

impl Iterator for Edges<'graph> {
    type Item = ConstraintIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.pointer {
            self.pointer = self.graph.next_constraints[p];
            Some(p)
        } else {
            None
        }
    }
}

crate struct RegionGraph<'s> {
    set: &'s ConstraintSet,
    constraint_graph: &'s ConstraintGraph,
}

impl RegionGraph<'s> {
    /// Create a "dependency graph" where each region constraint `R1:
    /// R2` is treated as an edge `R1 -> R2`. We use this graph to
    /// construct SCCs for region inference but also for error
    /// reporting.
    crate fn new(set: &'s ConstraintSet, constraint_graph: &'s ConstraintGraph) -> Self {
        Self {
            set,
            constraint_graph,
        }
    }

    /// Given a region `R`, iterate over all regions `R1` such that
    /// there exists a constraint `R: R1`.
    crate fn sub_regions(&self, region_sup: RegionVid) -> Successors<'_> {
        Successors {
            set: self.set,
            edges: self.constraint_graph.outgoing_edges(region_sup),
        }
    }
}

crate struct Successors<'s> {
    set: &'s ConstraintSet,
    edges: Edges<'s>,
}

impl Iterator for Successors<'s> {
    type Item = RegionVid;

    fn next(&mut self) -> Option<Self::Item> {
        self.edges.next().map(|c| self.set[c].sub)
    }
}

impl graph::DirectedGraph for RegionGraph<'_> {
    type Node = RegionVid;
}

impl graph::WithNumNodes for RegionGraph<'_> {
    fn num_nodes(&self) -> usize {
        self.constraint_graph.first_constraints.len()
    }
}

impl graph::WithSuccessors for RegionGraph<'_> {
    fn successors(
        &'graph self,
        node: Self::Node,
    ) -> <Self as graph::GraphSuccessors<'graph>>::Iter {
        self.sub_regions(node)
    }
}

impl graph::GraphSuccessors<'graph> for RegionGraph<'s> {
    type Item = RegionVid;
    type Iter = Successors<'graph>;
}
