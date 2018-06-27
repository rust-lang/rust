// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc_data_structures::fx::FxHashSet;
use rustc::ty::RegionVid;
use rustc::mir::Location;

use std::fmt;
use syntax_pos::Span;

#[derive(Clone, Default)]
crate struct ConstraintSet {
    constraints: IndexVec<ConstraintIndex, OutlivesConstraint>,
    seen_constraints: FxHashSet<(RegionVid, RegionVid)>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn push(&mut self, constraint: OutlivesConstraint) {
        debug!("add_outlives({:?}: {:?} @ {:?}",
               constraint.sup,
               constraint.sub,
               constraint.point);
        if constraint.sup == constraint.sub {
            // 'a: 'a is pretty uninteresting
            return;
        }
        if self.seen_constraints.insert(constraint.dedup_key()) {
            self.constraints.push(constraint);
        }
    }

    pub fn inner(&self) -> &IndexVec<ConstraintIndex, OutlivesConstraint> {
        &self.constraints
    }

    pub fn link(&mut self, len: usize) -> IndexVec<RegionVid, Option<ConstraintIndex>> {
        let mut map = IndexVec::from_elem_n(None, len);

        for (idx, constraint) in self.constraints.iter_enumerated_mut().rev() {
            let mut head = &mut map[constraint.sub];
            debug_assert!(constraint.next.is_none());
            constraint.next = *head;
            *head = Some(idx);
        }

        map
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

    /// At this location.
    pub point: Location,

    /// Later on, we thread the constraints onto a linked list
    /// grouped by their `sub` field. So if you had:
    ///
    /// Index | Constraint | Next Field
    /// ----- | ---------- | ----------
    /// 0     | `'a: 'b`   | Some(2)
    /// 1     | `'b: 'c`   | None
    /// 2     | `'c: 'b`   | None
    pub next: Option<ConstraintIndex>,

    /// Where did this constraint arise?
    pub span: Span,
}

impl OutlivesConstraint {
    pub fn dedup_key(&self) -> (RegionVid, RegionVid) {
        (self.sup, self.sub)
    }
}

impl fmt::Debug for OutlivesConstraint {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "({:?}: {:?} @ {:?}) due to {:?}",
            self.sup, self.sub, self.point, self.span
        )
    }
}

newtype_index!(ConstraintIndex { DEBUG_FORMAT = "ConstraintIndex({})" });

