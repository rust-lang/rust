use borrow_check::nll::region_infer::{ConstraintIndex, OutlivesConstraint};
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::FxHashSet;
use rustc::ty::RegionVid;

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

