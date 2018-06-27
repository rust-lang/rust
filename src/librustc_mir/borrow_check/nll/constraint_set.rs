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
    pub fn push(&mut self, outlives_constraint: OutlivesConstraint) {
        debug!("add_outlives({:?}: {:?} @ {:?}", outlives_constraint.sup, outlives_constraint.sub, outlives_constraint.point);
        if outlives_constraint.sup == outlives_constraint.sub {
            // 'a: 'a is pretty uninteresting
            return;
        }
        if self.seen_constraints.insert(outlives_constraint.dedup_key()) {
            self.constraints.push(outlives_constraint);
        }
    }

    pub fn iner(&self) -> &IndexVec<ConstraintIndex, OutlivesConstraint> {
        &self.constraints
    }

    /// Do Not use this to add nor remove items to the Vec, nor change the `sup`, nor `sub` of the data.
    pub fn iner_mut(&mut self) -> &mut IndexVec<ConstraintIndex, OutlivesConstraint> {
        &mut self.constraints
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

