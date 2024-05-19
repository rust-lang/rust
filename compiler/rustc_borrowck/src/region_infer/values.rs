use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::fx::FxIndexSet;
use rustc_index::bit_set::SparseBitMatrix;
use rustc_index::interval::IntervalSet;
use rustc_index::interval::SparseIntervalMatrix;
use rustc_index::Idx;
use rustc_middle::mir::{BasicBlock, Location};
use rustc_middle::ty::{self, RegionVid};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};
use std::fmt::Debug;
use std::rc::Rc;

use crate::BorrowIndex;

rustc_index::newtype_index! {
    /// A single integer representing a `ty::Placeholder`.
    #[debug_format = "PlaceholderIndex({})"]
    pub struct PlaceholderIndex {}
}

/// An individual element in a region value -- the value of a
/// particular region variable consists of a set of these elements.
#[derive(Debug, Clone)]
pub(crate) enum RegionElement {
    /// A point in the control-flow graph.
    Location(Location),

    /// A universally quantified region from the root universe (e.g.,
    /// a lifetime parameter).
    RootUniversalRegion(RegionVid),

    /// A placeholder (e.g., instantiated from a `for<'a> fn(&'a u32)`
    /// type).
    PlaceholderRegion(ty::PlaceholderRegion),
}

/// Records the CFG locations where each region is live. When we initially compute liveness, we use
/// an interval matrix storing liveness ranges for each region-vid.
pub(crate) struct LivenessValues {
    /// The map from locations to points.
    elements: Rc<DenseLocationMap>,

    /// Which regions are live. This is exclusive with the fine-grained tracking in `points`, and
    /// currently only used for validating promoteds (which don't care about more precise tracking).
    live_regions: Option<FxHashSet<RegionVid>>,

    /// For each region: the points where it is live.
    ///
    /// This is not initialized for promoteds, because we don't care *where* within a promoted a
    /// region is live, only that it is.
    points: Option<SparseIntervalMatrix<RegionVid, PointIndex>>,

    /// When using `-Zpolonius=next`, for each point: the loans flowing into the live regions at
    /// that point.
    pub(crate) loans: Option<LiveLoans>,
}

/// Data used to compute the loans that are live at a given point in the CFG, when using
/// `-Zpolonius=next`.
pub(crate) struct LiveLoans {
    /// The set of loans that flow into a given region. When individual regions are marked as live
    /// in the CFG, these inflowing loans are recorded as live.
    pub(crate) inflowing_loans: SparseBitMatrix<RegionVid, BorrowIndex>,

    /// The set of loans that are live at a given point in the CFG.
    pub(crate) live_loans: SparseBitMatrix<PointIndex, BorrowIndex>,
}

impl LiveLoans {
    pub(crate) fn new(num_loans: usize) -> Self {
        LiveLoans {
            live_loans: SparseBitMatrix::new(num_loans),
            inflowing_loans: SparseBitMatrix::new(num_loans),
        }
    }
}

impl LivenessValues {
    /// Create an empty map of regions to locations where they're live.
    pub(crate) fn with_specific_points(elements: Rc<DenseLocationMap>) -> Self {
        LivenessValues {
            live_regions: None,
            points: Some(SparseIntervalMatrix::new(elements.num_points())),
            elements,
            loans: None,
        }
    }

    /// Create an empty map of regions to locations where they're live.
    ///
    /// Unlike `with_specific_points`, does not track exact locations where something is live, only
    /// which regions are live.
    pub(crate) fn without_specific_points(elements: Rc<DenseLocationMap>) -> Self {
        LivenessValues {
            live_regions: Some(Default::default()),
            points: None,
            elements,
            loans: None,
        }
    }

    /// Iterate through each region that has a value in this set.
    pub(crate) fn regions(&self) -> impl Iterator<Item = RegionVid> + '_ {
        self.points.as_ref().expect("use with_specific_points").rows()
    }

    /// Iterate through each region that has a value in this set.
    // We are passing query instability implications to the caller.
    #[rustc_lint_query_instability]
    #[allow(rustc::potential_query_instability)]
    pub(crate) fn live_regions_unordered(&self) -> impl Iterator<Item = RegionVid> + '_ {
        self.live_regions.as_ref().unwrap().iter().copied()
    }

    /// Records `region` as being live at the given `location`.
    pub(crate) fn add_location(&mut self, region: RegionVid, location: Location) {
        let point = self.elements.point_from_location(location);
        debug!("LivenessValues::add_location(region={:?}, location={:?})", region, location);
        if let Some(points) = &mut self.points {
            points.insert(region, point);
        } else {
            if self.elements.point_in_range(point) {
                self.live_regions.as_mut().unwrap().insert(region);
            }
        }

        // When available, record the loans flowing into this region as live at the given point.
        if let Some(loans) = self.loans.as_mut() {
            if let Some(inflowing) = loans.inflowing_loans.row(region) {
                loans.live_loans.union_row(point, inflowing);
            }
        }
    }

    /// Records `region` as being live at all the given `points`.
    pub(crate) fn add_points(&mut self, region: RegionVid, points: &IntervalSet<PointIndex>) {
        debug!("LivenessValues::add_points(region={:?}, points={:?})", region, points);
        if let Some(this) = &mut self.points {
            this.union_row(region, points);
        } else {
            if points.iter().any(|point| self.elements.point_in_range(point)) {
                self.live_regions.as_mut().unwrap().insert(region);
            }
        }

        // When available, record the loans flowing into this region as live at the given points.
        if let Some(loans) = self.loans.as_mut() {
            if let Some(inflowing) = loans.inflowing_loans.row(region) {
                if !inflowing.is_empty() {
                    for point in points.iter() {
                        loans.live_loans.union_row(point, inflowing);
                    }
                }
            }
        }
    }

    /// Records `region` as being live at all the control-flow points.
    pub(crate) fn add_all_points(&mut self, region: RegionVid) {
        if let Some(points) = &mut self.points {
            points.insert_all_into_row(region);
        } else {
            self.live_regions.as_mut().unwrap().insert(region);
        }
    }

    /// Returns whether `region` is marked live at the given `location`.
    pub(crate) fn is_live_at(&self, region: RegionVid, location: Location) -> bool {
        let point = self.elements.point_from_location(location);
        if let Some(points) = &self.points {
            points.row(region).is_some_and(|r| r.contains(point))
        } else {
            unreachable!(
                "Should be using LivenessValues::with_specific_points to ask whether live at a location"
            )
        }
    }

    /// Returns an iterator of all the points where `region` is live.
    fn live_points(&self, region: RegionVid) -> impl Iterator<Item = PointIndex> + '_ {
        let Some(points) = &self.points else {
            unreachable!(
                "Should be using LivenessValues::with_specific_points to ask whether live at a location"
            )
        };
        points
            .row(region)
            .into_iter()
            .flat_map(|set| set.iter())
            .take_while(|&p| self.elements.point_in_range(p))
    }

    /// For debugging purposes, returns a pretty-printed string of the points where the `region` is
    /// live.
    pub(crate) fn pretty_print_live_points(&self, region: RegionVid) -> String {
        pretty_print_region_elements(
            self.live_points(region).map(|p| RegionElement::Location(self.elements.to_location(p))),
        )
    }

    #[inline]
    pub(crate) fn point_from_location(&self, location: Location) -> PointIndex {
        self.elements.point_from_location(location)
    }

    /// When using `-Zpolonius=next`, returns whether the `loan_idx` is live at the given `point`.
    pub(crate) fn is_loan_live_at(&self, loan_idx: BorrowIndex, point: PointIndex) -> bool {
        self.loans
            .as_ref()
            .expect("Accessing live loans requires `-Zpolonius=next`")
            .live_loans
            .contains(point, loan_idx)
    }
}

/// Maps from `ty::PlaceholderRegion` values that are used in the rest of
/// rustc to the internal `PlaceholderIndex` values that are used in
/// NLL.
#[derive(Debug, Default)]
pub(crate) struct PlaceholderIndices {
    indices: FxIndexSet<ty::PlaceholderRegion>,
}

impl PlaceholderIndices {
    /// Returns the `PlaceholderIndex` for the inserted `PlaceholderRegion`
    pub(crate) fn insert(&mut self, placeholder: ty::PlaceholderRegion) -> PlaceholderIndex {
        let (index, _) = self.indices.insert_full(placeholder);
        index.into()
    }

    pub(crate) fn lookup_index(&self, placeholder: ty::PlaceholderRegion) -> PlaceholderIndex {
        self.indices.get_index_of(&placeholder).unwrap().into()
    }

    pub(crate) fn lookup_placeholder(
        &self,
        placeholder: PlaceholderIndex,
    ) -> ty::PlaceholderRegion {
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
#[derive(Clone)]
pub(crate) struct RegionValues<N: Idx> {
    elements: Rc<DenseLocationMap>,
    placeholder_indices: Rc<PlaceholderIndices>,
    points: SparseIntervalMatrix<N, PointIndex>,
    free_regions: SparseBitMatrix<N, RegionVid>,

    /// Placeholders represent bound regions -- so something like `'a`
    /// in `for<'a> fn(&'a u32)`.
    placeholders: SparseBitMatrix<N, PlaceholderIndex>,
}

impl<N: Idx> RegionValues<N> {
    /// Creates a new set of "region values" that tracks causal information.
    /// Each of the regions in num_region_variables will be initialized with an
    /// empty set of points and no causal information.
    pub(crate) fn new(
        elements: &Rc<DenseLocationMap>,
        num_universal_regions: usize,
        placeholder_indices: &Rc<PlaceholderIndices>,
    ) -> Self {
        let num_placeholders = placeholder_indices.len();
        Self {
            elements: elements.clone(),
            points: SparseIntervalMatrix::new(elements.num_points()),
            placeholder_indices: placeholder_indices.clone(),
            free_regions: SparseBitMatrix::new(num_universal_regions),
            placeholders: SparseBitMatrix::new(num_placeholders),
        }
    }

    /// Adds the given element to the value for the given region. Returns whether
    /// the element is newly added (i.e., was not already present).
    pub(crate) fn add_element(&mut self, r: N, elem: impl ToElementIndex) -> bool {
        debug!("add(r={:?}, elem={:?})", r, elem);
        elem.add_to_row(self, r)
    }

    /// Adds all the control-flow points to the values for `r`.
    pub(crate) fn add_all_points(&mut self, r: N) {
        self.points.insert_all_into_row(r);
    }

    /// Adds all elements in `r_from` to `r_to` (because e.g., `r_to:
    /// r_from`).
    pub(crate) fn add_region(&mut self, r_to: N, r_from: N) -> bool {
        self.points.union_rows(r_from, r_to)
            | self.free_regions.union_rows(r_from, r_to)
            | self.placeholders.union_rows(r_from, r_to)
    }

    /// Returns `true` if the region `r` contains the given element.
    pub(crate) fn contains(&self, r: N, elem: impl ToElementIndex) -> bool {
        elem.contained_in_row(self, r)
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
        let block = self.elements.entry_point(block);
        let start = block.plus(start);
        let end = block.plus(end);
        let first_unset = row.first_unset_in(start..=end)?;
        Some(first_unset.index() - block.index())
    }

    /// `self[to] |= values[from]`, essentially: that is, take all the
    /// elements for the region `from` from `values` and add them to
    /// the region `to` in `self`.
    pub(crate) fn merge_liveness(&mut self, to: N, from: RegionVid, values: &LivenessValues) {
        let Some(value_points) = &values.points else {
            panic!("LivenessValues must track specific points for use in merge_liveness");
        };
        if let Some(set) = value_points.row(from) {
            self.points.union_row(to, set);
        }
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
    pub(crate) fn locations_outlived_by<'a>(&'a self, r: N) -> impl Iterator<Item = Location> + 'a {
        self.points.row(r).into_iter().flat_map(move |set| {
            set.iter()
                .take_while(move |&p| self.elements.point_in_range(p))
                .map(move |p| self.elements.to_location(p))
        })
    }

    /// Returns just the universal regions that are contained in a given region's value.
    pub(crate) fn universal_regions_outlived_by<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = RegionVid> + 'a {
        self.free_regions.row(r).into_iter().flat_map(|set| set.iter())
    }

    /// Returns all the elements contained in a given region's value.
    pub(crate) fn placeholders_contained_in<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = ty::PlaceholderRegion> + 'a {
        self.placeholders
            .row(r)
            .into_iter()
            .flat_map(|set| set.iter())
            .map(move |p| self.placeholder_indices.lookup_placeholder(p))
    }

    /// Returns all the elements contained in a given region's value.
    pub(crate) fn elements_contained_in<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = RegionElement> + 'a {
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
}

pub(crate) trait ToElementIndex: Debug + Copy {
    fn add_to_row<N: Idx>(self, values: &mut RegionValues<N>, row: N) -> bool;

    fn contained_in_row<N: Idx>(self, values: &RegionValues<N>, row: N) -> bool;
}

impl ToElementIndex for Location {
    fn add_to_row<N: Idx>(self, values: &mut RegionValues<N>, row: N) -> bool {
        let index = values.elements.point_from_location(self);
        values.points.insert(row, index)
    }

    fn contained_in_row<N: Idx>(self, values: &RegionValues<N>, row: N) -> bool {
        let index = values.elements.point_from_location(self);
        values.points.contains(row, index)
    }
}

impl ToElementIndex for RegionVid {
    fn add_to_row<N: Idx>(self, values: &mut RegionValues<N>, row: N) -> bool {
        values.free_regions.insert(row, self)
    }

    fn contained_in_row<N: Idx>(self, values: &RegionValues<N>, row: N) -> bool {
        values.free_regions.contains(row, self)
    }
}

impl ToElementIndex for ty::PlaceholderRegion {
    fn add_to_row<N: Idx>(self, values: &mut RegionValues<N>, row: N) -> bool {
        let index = values.placeholder_indices.lookup_index(self);
        values.placeholders.insert(row, index)
    }

    fn contained_in_row<N: Idx>(self, values: &RegionValues<N>, row: N) -> bool {
        let index = values.placeholder_indices.lookup_index(self);
        values.placeholders.contains(row, index)
    }
}

/// For debugging purposes, returns a pretty-printed string of the given points.
pub(crate) fn pretty_print_points(
    elements: &DenseLocationMap,
    points: impl IntoIterator<Item = PointIndex>,
) -> String {
    pretty_print_region_elements(
        points
            .into_iter()
            .take_while(|&p| elements.point_in_range(p))
            .map(|p| elements.to_location(p))
            .map(RegionElement::Location),
    )
}

/// For debugging purposes, returns a pretty-printed string of the given region elements.
fn pretty_print_region_elements(elements: impl IntoIterator<Item = RegionElement>) -> String {
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

    fn push_location_range(str: &mut String, location1: Location, location2: Location) {
        if location1 == location2 {
            str.push_str(&format!("{location1:?}"));
        } else {
            assert_eq!(location1.block, location2.block);
            str.push_str(&format!(
                "{:?}[{}..={}]",
                location1.block, location1.statement_index, location2.statement_index
            ));
        }
    }
}
