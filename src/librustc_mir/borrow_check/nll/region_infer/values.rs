// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::{BasicBlock, Location, Mir};
use rustc::ty::{self, RegionVid};
use rustc_data_structures::bit_set::{BitSet, SparseBitMatrix};
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::indexed_vec::IndexVec;
use std::fmt::Debug;
use std::rc::Rc;

/// Maps between a `Location` and a `PointIndex` (and vice versa).
crate struct RegionValueElements {
    /// For each basic block, how many points are contained within?
    statements_before_block: IndexVec<BasicBlock, usize>,

    /// Map backward from each point to the basic block that it
    /// belongs to.
    basic_blocks: IndexVec<PointIndex, BasicBlock>,

    num_points: usize,
}

impl RegionValueElements {
    crate fn new(mir: &Mir<'_>) -> Self {
        let mut num_points = 0;
        let statements_before_block: IndexVec<BasicBlock, usize> = mir
            .basic_blocks()
            .iter()
            .map(|block_data| {
                let v = num_points;
                num_points += block_data.statements.len() + 1;
                v
            })
            .collect();
        debug!(
            "RegionValueElements: statements_before_block={:#?}",
            statements_before_block
        );
        debug!("RegionValueElements: num_points={:#?}", num_points);

        let mut basic_blocks = IndexVec::with_capacity(num_points);
        for (bb, bb_data) in mir.basic_blocks().iter_enumerated() {
            basic_blocks.extend((0 .. bb_data.statements.len() + 1).map(|_| bb));
        }

        Self {
            statements_before_block,
            basic_blocks,
            num_points,
        }
    }

    /// Total number of point indices
    crate fn num_points(&self) -> usize {
        self.num_points
    }

    /// Converts a `Location` into a `PointIndex`. O(1).
    crate fn point_from_location(&self, location: Location) -> PointIndex {
        let Location {
            block,
            statement_index,
        } = location;
        let start_index = self.statements_before_block[block];
        PointIndex::new(start_index + statement_index)
    }

    /// Converts a `Location` into a `PointIndex`. O(1).
    crate fn entry_point(&self, block: BasicBlock) -> PointIndex {
        let start_index = self.statements_before_block[block];
        PointIndex::new(start_index)
    }

    /// Converts a `PointIndex` back to a location. O(1).
    crate fn to_location(&self, index: PointIndex) -> Location {
        assert!(index.index() < self.num_points);
        let block = self.basic_blocks[index];
        let start_index = self.statements_before_block[block];
        let statement_index = index.index() - start_index;
        Location { block, statement_index }
    }

    /// Sometimes we get point-indices back from bitsets that may be
    /// out of range (because they round up to the nearest 2^N number
    /// of bits). Use this function to filter such points out if you
    /// like.
    crate fn point_in_range(&self, index: PointIndex) -> bool {
        index.index() < self.num_points
    }

    /// Pushes all predecessors of `index` onto `stack`.
    crate fn push_predecessors(
        &self,
        mir: &Mir<'_>,
        index: PointIndex,
        stack: &mut Vec<PointIndex>,
    ) {
        let Location { block, statement_index } = self.to_location(index);
        if statement_index == 0 {
            // If this is a basic block head, then the predecessors are
            // the the terminators of other basic blocks
            stack.extend(
                mir
                    .predecessors_for(block)
                    .iter()
                    .map(|&pred_bb| mir.terminator_loc(pred_bb))
                    .map(|pred_loc| self.point_from_location(pred_loc)),
            );
        } else {
            // Otherwise, the pred is just the previous statement
            stack.push(PointIndex::new(index.index() - 1));
        }
    }
}

/// A single integer representing a `Location` in the MIR control-flow
/// graph. Constructed efficiently from `RegionValueElements`.
newtype_index! {
    pub struct PointIndex { DEBUG_FORMAT = "PointIndex({})" }
}

/// A single integer representing a (non-zero) `UniverseIndex`.
/// Computed just by subtracting one from `UniverseIndex`; this is
/// because the `0` value for `UniverseIndex` represents the root
/// universe, and we don't need/want a bit for that one.
newtype_index! {
    pub struct PlaceholderIndex { DEBUG_FORMAT = "PlaceholderIndex({})" }
}

/// An individual element in a region value -- the value of a
/// particular region variable consists of a set of these elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
crate enum RegionElement {
    /// A point in the control-flow graph.
    Location(Location),

    /// A universally quantified region from the root universe (e.g.,
    /// a lifetime parameter).
    RootUniversalRegion(RegionVid),

    /// A subuniverse from a subuniverse (e.g., instantiated from a
    /// `for<'a> fn(&'a u32)` type).
    SubUniversalRegion(ty::UniverseIndex),
}

/// When we initially compute liveness, we use a bit matrix storing
/// points for each region-vid.
crate struct LivenessValues<N: Idx> {
    elements: Rc<RegionValueElements>,
    points: SparseBitMatrix<N, PointIndex>,
}

impl<N: Idx> LivenessValues<N> {
    /// Creates a new set of "region values" that tracks causal information.
    /// Each of the regions in num_region_variables will be initialized with an
    /// empty set of points and no causal information.
    crate fn new(elements: &Rc<RegionValueElements>) -> Self {
        Self {
            elements: elements.clone(),
            points: SparseBitMatrix::new(elements.num_points),
        }
    }

    /// Iterate through each region that has a value in this set.
    crate fn rows<'a>(&'a self) -> impl Iterator<Item = N> {
        self.points.rows()
    }

    /// Adds the given element to the value for the given region. Returns true if
    /// the element is newly added (i.e., was not already present).
    crate fn add_element(&mut self, row: N, location: Location) -> bool {
        debug!("LivenessValues::add(r={:?}, location={:?})", row, location);
        let index = self.elements.point_from_location(location);
        self.points.insert(row, index)
    }

    /// Adds all the elements in the given bit array into the given
    /// region. Returns true if any of them are newly added.
    crate fn add_elements(&mut self, row: N, locations: &BitSet<PointIndex>) -> bool {
        debug!("LivenessValues::add_elements(row={:?}, locations={:?})", row, locations);
        self.points.union_into_row(row, locations)
    }

    /// Adds all the control-flow points to the values for `r`.
    crate fn add_all_points(&mut self, row: N) {
        self.points.insert_all_into_row(row);
    }

    /// True if the region `r` contains the given element.
    crate fn contains(&self, row: N, location: Location) -> bool {
        let index = self.elements.point_from_location(location);
        self.points.contains(row, index)
    }

    /// Returns a "pretty" string value of the region. Meant for debugging.
    crate fn region_value_str(&self, r: N) -> String {
        region_value_str(
            self.points
                .row(r)
                .into_iter()
                .flat_map(|set| set.iter())
                .take_while(|&p| self.elements.point_in_range(p))
                .map(|p| self.elements.to_location(p))
                .map(RegionElement::Location),
        )
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
crate struct RegionValues<N: Idx> {
    elements: Rc<RegionValueElements>,
    points: SparseBitMatrix<N, PointIndex>,
    free_regions: SparseBitMatrix<N, RegionVid>,

    /// Placeholders represent bound regions -- so something like `'a`
    /// in for<'a> fn(&'a u32)`.
    placeholders: SparseBitMatrix<N, PlaceholderIndex>,
}

impl<N: Idx> RegionValues<N> {
    /// Creates a new set of "region values" that tracks causal information.
    /// Each of the regions in num_region_variables will be initialized with an
    /// empty set of points and no causal information.
    crate fn new(
        elements: &Rc<RegionValueElements>,
        num_universal_regions: usize,
        max_universe: ty::UniverseIndex,
    ) -> Self {
        let num_placeholders = max_universe.as_usize();
        Self {
            elements: elements.clone(),
            points: SparseBitMatrix::new(elements.num_points),
            free_regions: SparseBitMatrix::new(num_universal_regions),
            placeholders: SparseBitMatrix::new(num_placeholders),
        }
    }

    /// Adds the given element to the value for the given region. Returns true if
    /// the element is newly added (i.e., was not already present).
    crate fn add_element(&mut self, r: N, elem: impl ToElementIndex) -> bool {
        debug!("add(r={:?}, elem={:?})", r, elem);
        elem.add_to_row(self, r)
    }

    /// Adds all the control-flow points to the values for `r`.
    crate fn add_all_points(&mut self, r: N) {
        self.points.insert_all_into_row(r);
    }

    /// Add all elements in `r_from` to `r_to` (because e.g. `r_to:
    /// r_from`).
    crate fn add_region(&mut self, r_to: N, r_from: N) -> bool {
        self.points.union_rows(r_from, r_to)
            | self.free_regions.union_rows(r_from, r_to)
            | self.placeholders.union_rows(r_from, r_to)
    }

    /// True if the region `r` contains the given element.
    crate fn contains(&self, r: N, elem: impl ToElementIndex) -> bool {
        elem.contained_in_row(self, r)
    }

    /// `self[to] |= values[from]`, essentially: that is, take all the
    /// elements for the region `from` from `values` and add them to
    /// the region `to` in `self`.
    crate fn merge_liveness<M: Idx>(&mut self, to: N, from: M, values: &LivenessValues<M>) {
        if let Some(set) = values.points.row(from) {
            self.points.union_into_row(to, set);
        }
    }

    /// True if `sup_region` contains all the CFG points that
    /// `sub_region` contains. Ignores universal regions.
    crate fn contains_points(&self, sup_region: N, sub_region: N) -> bool {
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
    crate fn locations_outlived_by<'a>(&'a self, r: N) -> impl Iterator<Item = Location> + 'a {
        self.points
            .row(r)
            .into_iter()
            .flat_map(move |set| {
                set.iter()
                    .take_while(move |&p| self.elements.point_in_range(p))
                    .map(move |p| self.elements.to_location(p))
            })
    }

    /// Returns just the universal regions that are contained in a given region's value.
    crate fn universal_regions_outlived_by<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = RegionVid> + 'a {
        self.free_regions
            .row(r)
            .into_iter()
            .flat_map(|set| set.iter())
    }

    /// Returns all the elements contained in a given region's value.
    crate fn subuniverses_contained_in<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = ty::UniverseIndex> + 'a {
        self.placeholders
            .row(r)
            .into_iter()
            .flat_map(|set| set.iter())
            .map(|p| ty::UniverseIndex::from_u32((p.index() + 1) as u32))
    }

    /// Returns all the elements contained in a given region's value.
    crate fn elements_contained_in<'a>(&'a self, r: N) -> impl Iterator<Item = RegionElement> + 'a {
        let points_iter = self.locations_outlived_by(r).map(RegionElement::Location);

        let free_regions_iter = self
            .universal_regions_outlived_by(r)
            .map(RegionElement::RootUniversalRegion);

        let subuniverses_iter = self
            .subuniverses_contained_in(r)
            .map(RegionElement::SubUniversalRegion);

        points_iter
            .chain(free_regions_iter)
            .chain(subuniverses_iter)
    }

    /// Returns a "pretty" string value of the region. Meant for debugging.
    crate fn region_value_str(&self, r: N) -> String {
        region_value_str(self.elements_contained_in(r))
    }
}

crate trait ToElementIndex: Debug + Copy {
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

impl ToElementIndex for ty::UniverseIndex {
    fn add_to_row<N: Idx>(self, values: &mut RegionValues<N>, row: N) -> bool {
        let index = PlaceholderIndex::new(self.as_usize() - 1);
        values.placeholders.insert(row, index)
    }

    fn contained_in_row<N: Idx>(self, values: &RegionValues<N>, row: N) -> bool {
        let index = PlaceholderIndex::new(self.as_usize() - 1);
        values.placeholders.contains(row, index)
    }
}

crate fn location_set_str(
    elements: &RegionValueElements,
    points: impl IntoIterator<Item = PointIndex>,
) -> String {
    region_value_str(
        points
            .into_iter()
            .take_while(|&p| elements.point_in_range(p))
            .map(|p| elements.to_location(p))
            .map(RegionElement::Location),
    )
}

fn region_value_str(elements: impl IntoIterator<Item = RegionElement>) -> String {
    let mut result = String::new();
    result.push_str("{");

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
                result.push_str(&format!("{:?}", fr));
            }

            RegionElement::SubUniversalRegion(ur) => {
                if let Some((location1, location2)) = open_location {
                    push_sep(&mut result);
                    push_location_range(&mut result, location1, location2);
                    open_location = None;
                }

                push_sep(&mut result);
                result.push_str(&format!("{:?}", ur));
            }
        }
    }

    if let Some((location1, location2)) = open_location {
        push_sep(&mut result);
        push_location_range(&mut result, location1, location2);
    }

    result.push_str("}");

    return result;

    fn push_location_range(str: &mut String, location1: Location, location2: Location) {
        if location1 == location2 {
            str.push_str(&format!("{:?}", location1));
        } else {
            assert_eq!(location1.block, location2.block);
            str.push_str(&format!(
                "{:?}[{}..={}]",
                location1.block, location1.statement_index, location2.statement_index
            ));
        }
    }
}
