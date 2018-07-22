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
use rustc::ty::RegionVid;
use rustc_data_structures::bitvec::SparseBitMatrix;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::indexed_vec::IndexVec;
use std::fmt::Debug;
use std::rc::Rc;
use std::ops::Range;

/// Maps between the various kinds of elements of a region value to
/// the internal indices that w use.
crate struct RegionValueElements {
    /// For each basic block, how many points are contained within?
    statements_before_block: IndexVec<BasicBlock, usize>,
    num_points: usize,
    num_universal_regions: usize,
}

impl RegionValueElements {
    crate fn new(mir: &Mir<'_>, num_universal_regions: usize) -> Self {
        let mut num_points = 0;
        let statements_before_block = mir
            .basic_blocks()
            .iter()
            .map(|block_data| {
                let v = num_points;
                num_points += block_data.statements.len() + 1;
                v
            })
            .collect();

        debug!(
            "RegionValueElements(num_universal_regions={:?})",
            num_universal_regions
        );
        debug!(
            "RegionValueElements: statements_before_block={:#?}",
            statements_before_block
        );
        debug!("RegionValueElements: num_points={:#?}", num_points);

        Self {
            statements_before_block,
            num_points,
            num_universal_regions,
        }
    }

    fn point_from_location(&self, location: Location) -> PointIndex {
        let Location {
            block,
            statement_index,
        } = location;
        let start_index = self.statements_before_block[block];
        PointIndex::new(start_index + statement_index)
    }

    /// Range coverting all point indices.
    fn all_points(&self) -> Range<PointIndex> {
        PointIndex::new(0)..PointIndex::new(self.num_points)
    }

    /// Converts a particular `RegionElementIndex` to a location, if
    /// that is what it represents. Returns `None` otherwise.
    crate fn to_location(&self, i: PointIndex) -> Location {
        let point_index = i.index();

        // Find the basic block. We have a vector with the
        // starting index of the statement in each block. Imagine
        // we have statement #22, and we have a vector like:
        //
        // [0, 10, 20]
        //
        // In that case, this represents point_index 2 of
        // basic block BB2. We know this because BB0 accounts for
        // 0..10, BB1 accounts for 11..20, and BB2 accounts for
        // 20...
        //
        // To compute this, we could do a binary search, but
        // because I am lazy we instead iterate through to find
        // the last point where the "first index" (0, 10, or 20)
        // was less than the statement index (22). In our case, this will
        // be (BB2, 20).
        //
        // Nit: we could do a binary search here but I'm too lazy.
        let (block, &first_index) = self
            .statements_before_block
            .iter_enumerated()
            .filter(|(_, first_index)| **first_index <= point_index)
            .last()
            .unwrap();

        Location {
            block,
            statement_index: point_index - first_index,
        }
    }
}

/// A single integer representing a `Location` in the MIR control-flow
/// graph. Constructed efficiently from `RegionValueElements`.
newtype_index!(PointIndex { DEBUG_FORMAT = "PointIndex({})" });

/// An individual element in a region value -- the value of a
/// particular region variable consists of a set of these elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
crate enum RegionElement {
    /// A point in the control-flow graph.
    Location(Location),

    /// A universally quantified region from the root universe (e.g.,
    /// a lifetime parameter).
    RootUniversalRegion(RegionVid),
}

/// Stores the values for a set of regions. These are stored in a
/// compact `SparseBitMatrix` representation, with one row per region
/// variable. The columns consist of either universal regions or
/// points in the CFG.
#[derive(Clone)]
crate struct RegionValues<N: Idx> {
    elements: Rc<RegionValueElements>,
    points: SparseBitMatrix<N, PointIndex>,
    free_regions: SparseBitMatrix<N, RegionVid>,
}

impl<N: Idx> RegionValues<N> {
    /// Creates a new set of "region values" that tracks causal information.
    /// Each of the regions in num_region_variables will be initialized with an
    /// empty set of points and no causal information.
    crate fn new(elements: &Rc<RegionValueElements>) -> Self {
        Self {
            elements: elements.clone(),
            points: SparseBitMatrix::new(elements.num_points),
            free_regions: SparseBitMatrix::new(elements.num_universal_regions),
        }
    }

    /// Adds the given element to the value for the given region. Returns true if
    /// the element is newly added (i.e., was not already present).
    crate fn add_element(
        &mut self,
        r: N,
        elem: impl ToElementIndex,
    ) -> bool {
        debug!("add(r={:?}, elem={:?})", r, elem);
        elem.add_to_row(self, r)
    }

    /// Adds all the control-flow points to the values for `r`.
    crate fn add_all_points(&mut self, r: N) {
        // FIXME OMG so inefficient. We'll fix later.
        for p in self.elements.all_points() {
            self.points.add(r, p);
        }
    }

    /// Add all elements in `r_from` to `r_to` (because e.g. `r_to:
    /// r_from`).
    crate fn add_region(&mut self, r_to: N, r_from: N) -> bool {
        self.points.merge(r_from, r_to) | self.free_regions.merge(r_from, r_to)
        // FIXME universes?
    }

    /// True if the region `r` contains the given element.
    crate fn contains(
        &self,
        r: N,
        elem: impl ToElementIndex,
    ) -> bool {
        elem.contained_in_row(self, r)
    }

    /// Iterate through each region that has a value in this set.
    crate fn regions_with_points<'a>(&'a self) -> impl Iterator<Item = N> {
        self.points.rows()
    }

    /// `self[to] |= values[from]`, essentially: that is, take all the
    /// elements for the region `from` from `values` and add them to
    /// the region `to` in `self`.
    crate fn merge_row<M: Idx>(&mut self, to: N, from: M, values: &RegionValues<M>) {
        if let Some(set) = values.points.row(from) {
            self.points.merge_into(to, set);
        }

        if let Some(set) = values.free_regions.row(from) {
            self.free_regions.merge_into(to, set);
        }
    }

    /// True if `sup_region` contains all the CFG points that
    /// `sub_region` contains. Ignores universal regions.
    crate fn contains_points(
        &self,
        sup_region: N,
        sub_region: N,
    ) -> bool {
        // This could be done faster by comparing the bitsets. But I
        // am lazy.
        if let Some(sub_row) = self.points.row(sub_region) {
            if let Some(sup_row) = self.points.row(sup_region) {
                sup_row.contains_all(sub_row)
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
    crate fn locations_outlived_by<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = Location> + 'a {
        self.points
            .row(r)
            .into_iter()
            .flat_map(move |set| set.iter().map(move |p| self.elements.to_location(p)))
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
    crate fn elements_contained_in<'a>(
        &'a self,
        r: N,
    ) -> impl Iterator<Item = RegionElement> + 'a {
        let points_iter = self
            .locations_outlived_by(r)
            .map(RegionElement::Location);

        let free_regions_iter = self
            .universal_regions_outlived_by(r)
            .map(RegionElement::RootUniversalRegion);

        points_iter.chain(free_regions_iter)
    }

    /// Returns a "pretty" string value of the region. Meant for debugging.
    crate fn region_value_str(&self, r: N) -> String {
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

        for element in self.elements_contained_in(r) {
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
                        Self::push_location_range(&mut result, location1, location2);
                    }

                    open_location = Some((l, l));
                }

                RegionElement::RootUniversalRegion(fr) => {
                    if let Some((location1, location2)) = open_location {
                        push_sep(&mut result);
                        Self::push_location_range(&mut result, location1, location2);
                        open_location = None;
                    }

                    push_sep(&mut result);
                    result.push_str(&format!("{:?}", fr));
                }
            }
        }

        if let Some((location1, location2)) = open_location {
            push_sep(&mut result);
            Self::push_location_range(&mut result, location1, location2);
        }

        result.push_str("}");

        result
    }

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

crate trait ToElementIndex: Debug + Copy {
    fn add_to_row<N: Idx>(
        self,
        values: &mut RegionValues<N>,
        row: N,
    ) -> bool;

    fn contained_in_row<N: Idx>(
        self,
        values: &RegionValues<N>,
        row: N,
    ) -> bool;
}

impl ToElementIndex for Location {
    fn add_to_row<N: Idx>(
        self,
        values: &mut RegionValues<N>,
        row: N,
    ) -> bool {
        let index = values.elements.point_from_location(self);
        values.points.add(row, index)
    }

    fn contained_in_row<N: Idx>(
        self,
        values: &RegionValues<N>,
        row: N,
    ) -> bool {
        let index = values.elements.point_from_location(self);
        values.points.contains(row, index)
    }
}

impl ToElementIndex for RegionVid {
    fn add_to_row<N: Idx>(
        self,
        values: &mut RegionValues<N>,
        row: N,
    ) -> bool {
        values.free_regions.add(row, self)
    }

    fn contained_in_row<N: Idx>(
        self,
        values: &RegionValues<N>,
        row: N,
    ) -> bool {
        values.free_regions.contains(row, self)
    }
}
