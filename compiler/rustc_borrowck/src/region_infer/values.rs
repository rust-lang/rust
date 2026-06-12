use std::fmt::Debug;
use std::rc::Rc;

use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_index::Idx;
use rustc_index::bit_set::SparseBitMatrix;
use rustc_index::interval::{IntervalSet, SparseIntervalMatrix};
use rustc_middle::bug;
use rustc_middle::mir::{BasicBlock, Location};
use rustc_middle::ty::{self, RegionVid};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};
use tracing::debug;

use crate::BorrowIndex;
use crate::polonius::LiveLoans;

rustc_index::newtype_index! {
    /// A single integer representing a `ty::Placeholder`.
    #[debug_format = "PlaceholderIndex({})"]
    pub(crate) struct PlaceholderIndex {}
}

/// An individual element in a region value -- the value of a
/// particular region variable consists of a set of these elements.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RegionElement<'tcx> {
    /// A point in the control-flow graph.
    Location(Location),

    /// A universally quantified region from the root universe (e.g.,
    /// a lifetime parameter).
    RootUniversalRegion(RegionVid),

    /// A placeholder (e.g., instantiated from a `for<'a> fn(&'a u32)`
    /// type).
    PlaceholderRegion(ty::PlaceholderRegion<'tcx>),
}

/// Either a mapping of which points a region is live at (for regular bodies),
/// or which regions are live in the body somewhere (for promoteds, which do
/// not care about where they are live, only that they are).
#[derive(Clone)] // FIXME(#146079)
enum LiveRegions {
    /// region `'r` is live at locations `L`.
    AtPoints(SparseIntervalMatrix<RegionVid, PointIndex>),
    /// Region `'r` is live in function body.
    InBody(FxHashSet<RegionVid>),
}

/// Records the CFG locations where each region is live. When we initially compute liveness, we use
/// an interval matrix storing liveness ranges for each region-vid.
#[derive(Clone)] // FIXME(#146079)
pub(crate) struct LivenessValues {
    /// The map from locations to points.
    location_map: Rc<DenseLocationMap>,

    /// Where a region is live.
    live_regions: LiveRegions,

    /// When using `-Zpolonius=next`, the set of loans that are live at a given point in the CFG.
    live_loans: Option<LiveLoans>,
}

impl LivenessValues {
    /// Create an empty map of regions to locations where they're live.
    pub(crate) fn with_specific_points(location_map: Rc<DenseLocationMap>) -> Self {
        LivenessValues {
            live_regions: LiveRegions::AtPoints(SparseIntervalMatrix::new(
                location_map.num_points(),
            )),
            location_map,
            live_loans: None,
        }
    }

    /// Create an empty map of regions to locations where they're live.
    ///
    /// Unlike `with_specific_points`, does not track exact locations where something is live, only
    /// which regions are live.
    pub(crate) fn without_specific_points(location_map: Rc<DenseLocationMap>) -> Self {
        LivenessValues {
            live_regions: LiveRegions::InBody(Default::default()),
            location_map,
            live_loans: None,
        }
    }

    /// Returns the liveness matrix of points where each region is live. Panics if the liveness
    /// values have been created without any per-point data (that is, for promoteds).
    pub(crate) fn points(&self) -> &SparseIntervalMatrix<RegionVid, PointIndex> {
        if let LiveRegions::AtPoints(points) = &self.live_regions {
            points
        } else {
            bug!("this `LivenessValues` wasn't created using `with_specific_points`")
        }
    }

    /// Get the liveness status of a region `r`, if any.
    /// Panics if liveness data is not tracked for any region.
    pub(crate) fn point_liveness(&self, region: RegionVid) -> Option<&IntervalSet<PointIndex>> {
        self.points().row(region)
    }

    /// Iterate through each region that has a value in this set.
    // We are passing query instability implications to the caller.
    #[rustc_lint_query_instability]
    #[allow(rustc::potential_query_instability)]
    pub(crate) fn live_regions_unordered(&self) -> impl Iterator<Item = RegionVid> {
        if let LiveRegions::InBody(live_regions) = &self.live_regions {
            live_regions.iter().copied()
        } else {
            bug!("this `LivenessValues` wasn't created using `without_specific_points`")
        }
    }

    /// Records `region` as being live at the given `location`.
    pub(crate) fn add_location(&mut self, region: RegionVid, location: Location) {
        let point = self.location_map.point_from_location(location);
        debug!("LivenessValues::add_location(region={:?}, location={:?})", region, location);
        match &mut self.live_regions {
            LiveRegions::AtPoints(points) => {
                points.insert(region, point);
            }

            LiveRegions::InBody(live_regions) if self.location_map.point_in_range(point) => {
                live_regions.insert(region);
            }

            LiveRegions::InBody(_) => (),
        };
    }

    /// Records `region` as being live at all the given `points`.
    pub(crate) fn add_points(&mut self, region: RegionVid, points: &IntervalSet<PointIndex>) {
        debug!("LivenessValues::add_points(region={:?}, points={:?})", region, points);
        match &mut self.live_regions {
            LiveRegions::AtPoints(these_points) => {
                these_points.union_row(region, points);
            }
            LiveRegions::InBody(live_regions)
                if points.iter().any(|point| self.location_map.point_in_range(point)) =>
            {
                live_regions.insert(region);
            }
            LiveRegions::InBody(_) => (),
        };
    }

    /// Records `region` as being live at all the control-flow points.
    pub(crate) fn add_all_points(&mut self, region: RegionVid) {
        match &mut self.live_regions {
            LiveRegions::AtPoints(points) => points.insert_all_into_row(region),
            LiveRegions::InBody(live_regions) => {
                live_regions.insert(region);
            }
        }
    }

    /// Returns whether `region` is marked live at the given
    /// [`location`][rustc_middle::mir::Location].
    pub(crate) fn is_live_at(&self, region: RegionVid, location: Location) -> bool {
        let point = self.location_map.point_from_location(location);
        self.is_live_at_point(region, point)
    }

    /// Returns whether `region` is marked live at the given
    /// [`point`][rustc_mir_dataflow::points::PointIndex].
    #[inline]
    pub(crate) fn is_live_at_point(&self, region: RegionVid, point: PointIndex) -> bool {
        self.point_liveness(region).is_some_and(|r| r.contains(point))
    }

    /// Returns an iterator of all the points where `region` is live.
    fn live_points(&self, region: RegionVid) -> impl Iterator<Item = PointIndex> {
        self.point_liveness(region)
            .into_iter()
            .flat_map(|set| set.iter())
            .take_while(|&p| self.location_map.point_in_range(p))
    }

    /// For debugging purposes, returns a pretty-printed string of the points where the `region` is
    /// live.
    pub(crate) fn pretty_print_live_points(&self, region: RegionVid) -> String {
        pretty_print_region_elements(
            self.live_points(region)
                .map(|p| RegionElement::Location(self.location_map.to_location(p))),
        )
    }

    #[inline]
    pub(crate) fn point_from_location(&self, location: Location) -> PointIndex {
        self.location_map.point_from_location(location)
    }

    #[inline]
    pub(crate) fn location_from_point(&self, point: PointIndex) -> Location {
        self.location_map.to_location(point)
    }

    /// When using `-Zpolonius=next`, records the given live loans for the loan scopes and active
    /// loans dataflow computations.
    pub(crate) fn record_live_loans(&mut self, live_loans: LiveLoans) {
        self.live_loans = Some(live_loans);
    }

    /// When using `-Zpolonius=next`, returns whether the `loan_idx` is live at the given `point`.
    pub(crate) fn is_loan_live_at(&self, loan_idx: BorrowIndex, point: PointIndex) -> bool {
        self.live_loans
            .as_ref()
            .expect("Accessing live loans requires `-Zpolonius=next`")
            .contains(point, loan_idx)
    }
}

/// Maps from `ty::PlaceholderRegion` values that are used in the rest of
/// rustc to the internal `PlaceholderIndex` values that are used in
/// NLL.
#[derive(Debug, Default)]
#[derive(Clone)] // FIXME(#146079)
pub(crate) struct PlaceholderIndices<'tcx> {
    indices: FxIndexSet<ty::PlaceholderRegion<'tcx>>,
}

impl<'tcx> PlaceholderIndices<'tcx> {
    /// Returns the `PlaceholderIndex` for the inserted `PlaceholderRegion`
    pub(crate) fn insert(&mut self, placeholder: ty::PlaceholderRegion<'tcx>) -> PlaceholderIndex {
        let (index, _) = self.indices.insert_full(placeholder);
        index.into()
    }

    pub(crate) fn lookup_index(
        &self,
        placeholder: ty::PlaceholderRegion<'tcx>,
    ) -> PlaceholderIndex {
        self.indices.get_index_of(&placeholder).unwrap().into()
    }

    pub(crate) fn lookup_placeholder(
        &self,
        placeholder: PlaceholderIndex,
    ) -> ty::PlaceholderRegion<'tcx> {
        self.indices[placeholder.index()]
    }

    pub(crate) fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Stores the full values for a set of regions (in contrast to
/// `LivenessValues`, which only stores those points in the where a
/// region is live). The full value for a region may contain points in
/// the CFG, but also free regions as well as bound universe
/// placeholders.
///
/// Example:
///
/// ```text
/// fn foo(x: &'a u32) -> &'a u32 {
///    let y: &'0 u32 = x; // let's call this `'0`
///    y
/// }
/// ```
///
/// Here, the variable `'0` would contain the free region `'a`,
/// because (since it is returned) it must live for at least `'a`. But
/// it would also contain various points from within the function.
pub(crate) struct RegionValues<'tcx, N: Idx> {
    location_map: Rc<DenseLocationMap>,
    placeholder_indices: PlaceholderIndices<'tcx>,
    points: SparseIntervalMatrix<N, PointIndex>,
    free_regions: SparseBitMatrix<N, RegionVid>,

    /// Placeholders represent bound regions -- so something like `'a`
    /// in `for<'a> fn(&'a u32)`.
    placeholders: SparseBitMatrix<N, PlaceholderIndex>,
}

impl<'tcx, N: Idx> RegionValues<'tcx, N> {
    /// Creates a new set of "region values" that tracks causal information.
    /// Each of the regions in num_region_variables will be initialized with an
    /// empty set of points and no causal information.
    pub(crate) fn new(
        location_map: Rc<DenseLocationMap>,
        num_universal_regions: usize,
        placeholder_indices: PlaceholderIndices<'tcx>,
    ) -> Self {
        let num_points = location_map.num_points();
        let num_placeholders = placeholder_indices.len();
        Self {
            location_map,
            points: SparseIntervalMatrix::new(num_points),
            placeholder_indices,
            free_regions: SparseBitMatrix::new(num_universal_regions),
            placeholders: SparseBitMatrix::new(num_placeholders),
        }
    }

    /// Adds all elements in `r_from` to `r_to` (because e.g., `r_to:
    /// r_from`).
    pub(crate) fn add_region(&mut self, r_to: N, r_from: N) -> bool {
        self.points.union_rows(r_from, r_to)
            | self.free_regions.union_rows(r_from, r_to)
            | self.placeholders.union_rows(r_from, r_to)
    }

    /// Returns the lowest statement index in `start..=end` which is not contained by `r`.
    pub(crate) fn first_non_contained_inclusive(
        &self,
        r: N,
        block: BasicBlock,
        start: usize,
        end: usize,
    ) -> Option<usize> {
        let row = self.points.row(r)?;
        let block = self.location_map.entry_point(block);
        let start = block.plus(start);
        let end = block.plus(end);
        let first_unset = row.first_unset_in(start..=end)?;
        Some(first_unset.index() - block.index())
    }

    /// Merge a row of liveness into our points.
    pub(crate) fn merge_liveness(&mut self, to: N, liveness: &IntervalSet<PointIndex>) {
        self.points.union_row(to, liveness);
    }

    /// Returns `true` if `sup_region` contains all the CFG points that
    /// `sub_region` contains. Ignores universal regions.
    pub(crate) fn contains_points(&self, sup_region: N, sub_region: N) -> bool {
        if let Some(sub_row) = self.points.row(sub_region) {
            if let Some(sup_row) = self.points.row(sup_region) {
                sup_row.superset(sub_row)
            } else {
                // sup row is empty, so sub row must be empty
                sub_row.is_empty()
            }
        } else {
            // sub row is empty, always true
            true
        }
    }

    /// Returns the locations contained within a given region `r`.
    pub(crate) fn locations_outlived_by(&self, r: N) -> impl Iterator<Item = Location> {
        self.points.row(r).into_iter().flat_map(move |set| {
            set.iter()
                .take_while(move |&p| self.location_map.point_in_range(p))
                .map(move |p| self.location_map.to_location(p))
        })
    }

    /// Returns just the universal regions that are contained in a given region's value.
    pub(crate) fn universal_regions_outlived_by(&self, r: N) -> impl Iterator<Item = RegionVid> {
        self.free_regions.row(r).into_iter().flat_map(|set| set.iter())
    }

    /// Returns all the elements contained in a given region's value.
    pub(crate) fn placeholders_contained_in(
        &self,
        r: N,
    ) -> impl Iterator<Item = ty::PlaceholderRegion<'tcx>> {
        self.placeholders
            .row(r)
            .into_iter()
            .flat_map(|set| set.iter())
            .map(move |p| self.placeholder_indices.lookup_placeholder(p))
    }

    /// Returns all the elements contained in a given region's value.
    pub(crate) fn elements_contained_in(&self, r: N) -> impl Iterator<Item = RegionElement<'tcx>> {
        let points_iter = self.locations_outlived_by(r).map(RegionElement::Location);

        let free_regions_iter =
            self.universal_regions_outlived_by(r).map(RegionElement::RootUniversalRegion);

        let placeholder_universes_iter =
            self.placeholders_contained_in(r).map(RegionElement::PlaceholderRegion);

        points_iter.chain(free_regions_iter).chain(placeholder_universes_iter)
    }

    /// Returns a "pretty" string value of the region. Meant for debugging.
    pub(crate) fn region_value_str(&self, r: N) -> String {
        pretty_print_region_elements(self.elements_contained_in(r))
    }

    /// Add a the free region with rvid `region` to SCC `scc`
    pub(crate) fn add_free_region(&mut self, scc: N, region: RegionVid) {
        self.free_regions.insert(scc, region);
    }

    pub(crate) fn add_placeholder(&mut self, scc: N, placeholder: ty::PlaceholderRegion<'tcx>) {
        let index = self.placeholder_indices.lookup_index(placeholder);
        self.placeholders.insert(scc, index);
    }

    /// Determine if `scc` contains the CFG point `p`.
    pub(crate) fn contains_point(&self, scc: N, p: Location) -> bool {
        let index = self.location_map.point_from_location(p);
        self.points.contains(scc, index)
    }

    /// Determine if `scc` contains the free region `free_region`.
    pub(crate) fn contains_free_region(&self, scc: N, free_region: RegionVid) -> bool {
        self.free_regions.contains(scc, free_region)
    }
}

/// For debugging purposes, returns a pretty-printed string of the given points.
pub(crate) fn pretty_print_points(
    location_map: &DenseLocationMap,
    points: impl IntoIterator<Item = PointIndex>,
) -> String {
    pretty_print_region_elements(
        points
            .into_iter()
            .take_while(|&p| location_map.point_in_range(p))
            .map(|p| location_map.to_location(p))
            .map(RegionElement::Location),
    )
}

/// For debugging purposes, returns a pretty-printed string of the given region elements.
fn pretty_print_region_elements<'tcx>(
    elements: impl IntoIterator<Item = RegionElement<'tcx>>,
) -> String {
    let mut result = String::new();
    result.push('{');

    // Set to Some(l1, l2) when we have observed all the locations
    // from l1..=l2 (inclusive) but not yet printed them. This
    // gets extended if we then see l3 where l3 is the successor
    // to l2.
    let mut open_location: Option<(Location, Location)> = None;

    let mut sep = "";
    let mut push_sep = |s: &mut String| {
        s.push_str(sep);
        sep = ", ";
    };

    for element in elements {
        match element {
            RegionElement::Location(l) => {
                if let Some((location1, location2)) = open_location {
                    if location2.block == l.block
                        && location2.statement_index == l.statement_index - 1
                    {
                        open_location = Some((location1, l));
                        continue;
                    }

                    push_sep(&mut result);
                    push_location_range(&mut result, location1, location2);
                }

                open_location = Some((l, l));
            }

            RegionElement::RootUniversalRegion(fr) => {
                if let Some((location1, location2)) = open_location {
                    push_sep(&mut result);
                    push_location_range(&mut result, location1, location2);
                    open_location = None;
                }

                push_sep(&mut result);
                result.push_str(&format!("{fr:?}"));
            }

            RegionElement::PlaceholderRegion(placeholder) => {
                if let Some((location1, location2)) = open_location {
                    push_sep(&mut result);
                    push_location_range(&mut result, location1, location2);
                    open_location = None;
                }

                push_sep(&mut result);
                result.push_str(&format!("{placeholder:?}"));
            }
        }
    }

    if let Some((location1, location2)) = open_location {
        push_sep(&mut result);
        push_location_range(&mut result, location1, location2);
    }

    result.push('}');

    return result;

    fn push_location_range(s: &mut String, location1: Location, location2: Location) {
        if location1 == location2 {
            s.push_str(&format!("{location1:?}"));
        } else {
            assert_eq!(location1.block, location2.block);
            s.push_str(&format!(
                "{:?}[{}..={}]",
                location1.block, location1.statement_index, location2.statement_index
            ));
        }
    }
}
