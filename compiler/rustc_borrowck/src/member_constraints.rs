use rustc_data_structures::fx::FxHashMap;
use rustc_index::vec::IndexVec;
use rustc_middle::infer::MemberConstraint;
use rustc_middle::ty::{self, Ty};
use rustc_span::Span;
use std::hash::Hash;
use std::ops::Index;

/// Compactly stores a set of `R0 member of [R1...Rn]` constraints,
/// indexed by the region `R0`.
crate struct MemberConstraintSet<'tcx, R>
where
    R: Copy + Eq,
{
    /// Stores the first "member" constraint for a given `R0`. This is an
    /// index into the `constraints` vector below.
    first_constraints: FxHashMap<R, NllMemberConstraintIndex>,

    /// Stores the data about each `R0 member of [R1..Rn]` constraint.
    /// These are organized into a linked list, so each constraint
    /// contains the index of the next constraint with the same `R0`.
    constraints: IndexVec<NllMemberConstraintIndex, NllMemberConstraint<'tcx>>,

    /// Stores the `R1..Rn` regions for *all* sets. For any given
    /// constraint, we keep two indices so that we can pull out a
    /// slice.
    choice_regions: Vec<ty::RegionVid>,
}

/// Represents a `R0 member of [R1..Rn]` constraint
crate struct NllMemberConstraint<'tcx> {
    next_constraint: Option<NllMemberConstraintIndex>,

    /// The span where the hidden type was instantiated.
    crate definition_span: Span,

    /// The hidden type in which `R0` appears. (Used in error reporting.)
    crate hidden_ty: Ty<'tcx>,

    /// The region `R0`.
    crate member_region_vid: ty::RegionVid,

    /// Index of `R1` in `choice_regions` vector from `MemberConstraintSet`.
    start_index: usize,

    /// Index of `Rn` in `choice_regions` vector from `MemberConstraintSet`.
    end_index: usize,
}

rustc_index::newtype_index! {
    crate struct NllMemberConstraintIndex {
        DEBUG_FORMAT = "MemberConstraintIndex({})"
    }
}

impl Default for MemberConstraintSet<'tcx, ty::RegionVid> {
    fn default() -> Self {
        Self {
            first_constraints: Default::default(),
            constraints: Default::default(),
            choice_regions: Default::default(),
        }
    }
}

impl<'tcx> MemberConstraintSet<'tcx, ty::RegionVid> {
    /// Pushes a member constraint into the set.
    ///
    /// The input member constraint `m_c` is in the form produced by
    /// the `rustc_middle::infer` code.
    ///
    /// The `to_region_vid` callback fn is used to convert the regions
    /// within into `RegionVid` format -- it typically consults the
    /// `UniversalRegions` data structure that is known to the caller
    /// (but which this code is unaware of).
    crate fn push_constraint(
        &mut self,
        m_c: &MemberConstraint<'tcx>,
        mut to_region_vid: impl FnMut(ty::Region<'tcx>) -> ty::RegionVid,
    ) {
        debug!("push_constraint(m_c={:?})", m_c);
        let member_region_vid: ty::RegionVid = to_region_vid(m_c.member_region);
        let next_constraint = self.first_constraints.get(&member_region_vid).cloned();
        let start_index = self.choice_regions.len();
        let end_index = start_index + m_c.choice_regions.len();
        debug!("push_constraint: member_region_vid={:?}", member_region_vid);
        let constraint_index = self.constraints.push(NllMemberConstraint {
            next_constraint,
            member_region_vid,
            definition_span: m_c.definition_span,
            hidden_ty: m_c.hidden_ty,
            start_index,
            end_index,
        });
        self.first_constraints.insert(member_region_vid, constraint_index);
        self.choice_regions.extend(m_c.choice_regions.iter().map(|&r| to_region_vid(r)));
    }
}

impl<R1> MemberConstraintSet<'tcx, R1>
where
    R1: Copy + Hash + Eq,
{
    /// Remap the "member region" key using `map_fn`, producing a new
    /// member constraint set.  This is used in the NLL code to map from
    /// the original `RegionVid` to an scc index. In some cases, we
    /// may have multiple `R1` values mapping to the same `R2` key -- that
    /// is ok, the two sets will be merged.
    crate fn into_mapped<R2>(
        self,
        mut map_fn: impl FnMut(R1) -> R2,
    ) -> MemberConstraintSet<'tcx, R2>
    where
        R2: Copy + Hash + Eq,
    {
        // We can re-use most of the original data, just tweaking the
        // linked list links a bit.
        //
        // For example if we had two keys `Ra` and `Rb` that both now
        // wind up mapped to the same key `S`, we would append the
        // linked list for `Ra` onto the end of the linked list for
        // `Rb` (or vice versa) -- this basically just requires
        // rewriting the final link from one list to point at the other
        // other (see `append_list`).

        let MemberConstraintSet { first_constraints, mut constraints, choice_regions } = self;

        let mut first_constraints2 = FxHashMap::default();
        first_constraints2.reserve(first_constraints.len());

        for (r1, start1) in first_constraints {
            let r2 = map_fn(r1);
            if let Some(&start2) = first_constraints2.get(&r2) {
                append_list(&mut constraints, start1, start2);
            }
            first_constraints2.insert(r2, start1);
        }

        MemberConstraintSet { first_constraints: first_constraints2, constraints, choice_regions }
    }
}

impl<R> MemberConstraintSet<'tcx, R>
where
    R: Copy + Hash + Eq,
{
    crate fn all_indices(&self) -> impl Iterator<Item = NllMemberConstraintIndex> {
        self.constraints.indices()
    }

    /// Iterate down the constraint indices associated with a given
    /// peek-region.  You can then use `choice_regions` and other
    /// methods to access data.
    crate fn indices(
        &self,
        member_region_vid: R,
    ) -> impl Iterator<Item = NllMemberConstraintIndex> + '_ {
        let mut next = self.first_constraints.get(&member_region_vid).cloned();
        std::iter::from_fn(move || -> Option<NllMemberConstraintIndex> {
            if let Some(current) = next {
                next = self.constraints[current].next_constraint;
                Some(current)
            } else {
                None
            }
        })
    }

    /// Returns the "choice regions" for a given member
    /// constraint. This is the `R1..Rn` from a constraint like:
    ///
    /// ```
    /// R0 member of [R1..Rn]
    /// ```
    crate fn choice_regions(&self, pci: NllMemberConstraintIndex) -> &[ty::RegionVid] {
        let NllMemberConstraint { start_index, end_index, .. } = &self.constraints[pci];
        &self.choice_regions[*start_index..*end_index]
    }
}

impl<'tcx, R> Index<NllMemberConstraintIndex> for MemberConstraintSet<'tcx, R>
where
    R: Copy + Eq,
{
    type Output = NllMemberConstraint<'tcx>;

    fn index(&self, i: NllMemberConstraintIndex) -> &NllMemberConstraint<'tcx> {
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
    constraints: &mut IndexVec<NllMemberConstraintIndex, NllMemberConstraint<'_>>,
    target_list: NllMemberConstraintIndex,
    source_list: NllMemberConstraintIndex,
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
