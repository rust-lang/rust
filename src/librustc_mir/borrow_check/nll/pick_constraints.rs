use crate::rustc::ty::{self, Ty};
use rustc::hir::def_id::DefId;
use rustc::infer::region_constraints::PickConstraint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use std::hash::Hash;
use std::ops::Index;
use syntax_pos::Span;

/// Compactly stores a set of `pick R0 in [R1...Rn]` constraints,
/// indexed by the region R0.
crate struct PickConstraintSet<'tcx, R>
where
    R: Copy + Hash + Eq,
{
    /// Stores the first "pick" constraint for a given R0. This is an
    /// index into the `constraints` vector below.
    first_constraints: FxHashMap<R, NllPickConstraintIndex>,

    /// Stores the data about each `pick R0 from [R1..Rn]` constraint.
    /// These are organized into a linked list, so each constraint
    /// contains the index of the next constraint with the same R0.
    constraints: IndexVec<NllPickConstraintIndex, NllPickConstraint<'tcx>>,

    /// Stores the `R1..Rn` regions for *all* sets. For any given
    /// constraint, we keep two indices so that we can pull out a
    /// slice.
    option_regions: Vec<ty::RegionVid>,
}

/// Represents a `pick R0 in [R1..Rn]` constraint
crate struct NllPickConstraint<'tcx> {
    next_constraint: Option<NllPickConstraintIndex>,

    /// The opaque type whose hidden type is being inferred. (Used in error reporting.)
    crate opaque_type_def_id: DefId,

    /// The span where the hidden type was instantiated.
    crate definition_span: Span,

    /// The hidden type in which R0 appears. (Used in error reporting.)
    crate hidden_ty: Ty<'tcx>,

    /// The region R0.
    crate pick_region_vid: ty::RegionVid,

    /// Index of `R1` in `option_regions` vector from `PickConstraintSet`.
    start_index: usize,

    /// Index of `Rn` in `option_regions` vector from `PickConstraintSet`.
    end_index: usize,
}

newtype_index! {
    crate struct NllPickConstraintIndex {
        DEBUG_FORMAT = "PickConstraintIndex({})"
    }
}

impl Default for PickConstraintSet<'tcx, ty::RegionVid> {
    fn default() -> Self {
        Self {
            first_constraints: Default::default(),
            constraints: Default::default(),
            option_regions: Default::default(),
        }
    }
}

impl<'tcx> PickConstraintSet<'tcx, ty::RegionVid> {
    crate fn push_constraint(
        &mut self,
        p_c: &PickConstraint<'tcx>,
        mut to_region_vid: impl FnMut(ty::Region<'tcx>) -> ty::RegionVid,
    ) {
        debug!("push_constraint(p_c={:?})", p_c);
        let pick_region_vid: ty::RegionVid = to_region_vid(p_c.pick_region);
        let next_constraint = self.first_constraints.get(&pick_region_vid).cloned();
        let start_index = self.option_regions.len();
        let end_index = start_index + p_c.option_regions.len();
        debug!("push_constraint: pick_region_vid={:?}", pick_region_vid);
        let constraint_index = self.constraints.push(NllPickConstraint {
            next_constraint,
            pick_region_vid,
            opaque_type_def_id: p_c.opaque_type_def_id,
            definition_span: p_c.definition_span,
            hidden_ty: p_c.hidden_ty,
            start_index,
            end_index,
        });
        self.first_constraints.insert(pick_region_vid, constraint_index);
        self.option_regions.extend(p_c.option_regions.iter().map(|&r| to_region_vid(r)));
    }
}

impl<'tcx, R1> PickConstraintSet<'tcx, R1>
where
    R1: Copy + Hash + Eq,
{
    /// Remap the "pick region" key using `map_fn`, producing a new
    /// pick-constraint set.  This is used in the NLL code to map from
    /// the original `RegionVid` to an scc index. In some cases, we
    /// may have multiple R1 values mapping to the same R2 key -- that
    /// is ok, the two sets will be merged.
    crate fn into_mapped<R2>(self, mut map_fn: impl FnMut(R1) -> R2) -> PickConstraintSet<'tcx, R2>
    where
        R2: Copy + Hash + Eq,
    {
        // We can re-use most of the original data, just tweaking the
        // linked list links a bit.
        //
        // For example if we had two keys Ra and Rb that both now wind
        // up mapped to the same key S, we would append the linked
        // list for Ra onto the end of the linked list for Rb (or vice
        // versa) -- this basically just requires rewriting the final
        // link from one list to point at the othe other (see
        // `append_list`).

        let PickConstraintSet { first_constraints, mut constraints, option_regions } = self;

        let mut first_constraints2 = FxHashMap::default();
        first_constraints2.reserve(first_constraints.len());

        for (r1, start1) in first_constraints {
            let r2 = map_fn(r1);
            if let Some(&start2) = first_constraints2.get(&r2) {
                append_list(&mut constraints, start1, start2);
            }
            first_constraints2.insert(r2, start1);
        }

        PickConstraintSet {
            first_constraints: first_constraints2,
            constraints,
            option_regions,
        }
    }
}

impl<'tcx, R> PickConstraintSet<'tcx, R>
where
    R: Copy + Hash + Eq,
{
    crate fn all_indices(
        &self,
    ) -> impl Iterator<Item = NllPickConstraintIndex> {
        self.constraints.indices()
    }

    /// Iterate down the constraint indices associated with a given
    /// peek-region.  You can then use `option_regions` and other
    /// methods to access data.
    crate fn indices(
        &self,
        pick_region_vid: R,
    ) -> impl Iterator<Item = NllPickConstraintIndex> + '_ {
        let mut next = self.first_constraints.get(&pick_region_vid).cloned();
        std::iter::from_fn(move || -> Option<NllPickConstraintIndex> {
            if let Some(current) = next {
                next = self.constraints[current].next_constraint;
                Some(current)
            } else {
                None
            }
        })
    }

    /// Returns the "option regions" for a given pick constraint. This is the R1..Rn from
    /// a constraint like:
    ///
    /// ```
    /// pick R0 in [R1..Rn]
    /// ```
    crate fn option_regions(&self, pci: NllPickConstraintIndex) -> &[ty::RegionVid] {
        let NllPickConstraint { start_index, end_index, .. } = &self.constraints[pci];
        &self.option_regions[*start_index..*end_index]
    }
}

impl<'tcx, R> Index<NllPickConstraintIndex> for PickConstraintSet<'tcx, R>
where
    R: Copy + Hash + Eq,
{
    type Output = NllPickConstraint<'tcx>;

    fn index(&self, i: NllPickConstraintIndex) -> &NllPickConstraint<'tcx> {
        &self.constraints[i]
    }
}

/// Given a linked list starting at `source_list` and another linked
/// list starting at `target_list`, modify `target_list` so that it is
/// followed by `source_list`.
///
/// Before:
///
/// ```
/// target_list: A -> B -> C -> (None)
/// source_list: D -> E -> F -> (None)
/// ```
///
/// After:
///
/// ```
/// target_list: A -> B -> C -> D -> E -> F -> (None)
/// ```
fn append_list(
    constraints: &mut IndexVec<NllPickConstraintIndex, NllPickConstraint<'_>>,
    target_list: NllPickConstraintIndex,
    source_list: NllPickConstraintIndex,
) {
    let mut p = target_list;
    loop {
        let mut r = &mut constraints[p];
        match r.next_constraint {
            Some(q) => p = q,
            None => {
                r.next_constraint = Some(source_list);
                return;
            }
        }
    }
}
