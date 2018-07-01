// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::RegionVid;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use borrow_check::nll::type_check::Locations;

use std::fmt;
use std::ops::Deref;

#[derive(Clone, Default)]
crate struct ConstraintSet {
    constraints: IndexVec<ConstraintIndex, OutlivesConstraint>,
}

impl ConstraintSet {
    pub fn push(&mut self, constraint: OutlivesConstraint) {
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
}

impl Deref for ConstraintSet {
    type Target = IndexVec<ConstraintIndex, OutlivesConstraint>;

    fn deref(&self) -> &Self::Target { &self.constraints }
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

newtype_index!(ConstraintIndex { DEBUG_FORMAT = "ConstraintIndex({})" });

crate struct ConstraintGraph {
    first_constraints: IndexVec<RegionVid, Option<ConstraintIndex>>,
    next_constraints: IndexVec<ConstraintIndex, Option<ConstraintIndex>>,
}

impl ConstraintGraph {
    /// Constraint a graph where each region constraint `R1: R2` is
    /// treated as an edge `R2 -> R1`. This is useful for cheaply
    /// finding dirty constraints.
    crate fn new(set: &ConstraintSet, num_region_vars: usize) -> Self {
        let mut first_constraints = IndexVec::from_elem_n(None, num_region_vars);
        let mut next_constraints = IndexVec::from_elem(None, &set.constraints);

        for (idx, constraint) in set.constraints.iter_enumerated().rev() {
            let mut head = &mut first_constraints[constraint.sub];
            let mut next = &mut next_constraints[idx];
            debug_assert!(next.is_none());
            *next = *head;
            *head = Some(idx);
        }

        ConstraintGraph { first_constraints, next_constraints }
    }

    /// Invokes `op` with the index of any constraints of the form
    /// `region_sup: region_sub`.  These are the constraints that must
    /// be reprocessed when the value of `R1` changes. If you think of
    /// each constraint `R1: R2` as an edge `R2 -> R1`, then this
    /// gives the set of successors to R2.
    crate fn for_each_dependent(
        &self,
        region_sub: RegionVid,
        mut op: impl FnMut(ConstraintIndex),
    ) {
        let mut p = self.first_constraints[region_sub];
        while let Some(dep_idx) = p {
            op(dep_idx);
            p = self.next_constraints[dep_idx];
        }
    }
}

