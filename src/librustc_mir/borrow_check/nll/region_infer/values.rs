// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;
use rustc_data_structures::bitvec::BitMatrix;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc::mir::{BasicBlock, Location, Mir};
use rustc::ty::RegionVid;

/// Maps between the various kinds of elements of a region value to
/// the internal indices that w use.
pub(super) struct RegionValueElements {
    /// For each basic block, how many points are contained within?
    statements_before_block: IndexVec<BasicBlock, usize>,
    num_points: usize,
    num_universal_regions: usize,
}

impl RegionValueElements {
    pub(super) fn new(mir: &Mir<'_>, num_universal_regions: usize) -> Self {
        let mut num_points = 0;
        let statements_before_block =
            mir.basic_blocks()
               .iter()
               .map(|block_data| {
                   let v = num_points;
                   num_points += block_data.statements.len() + 1;
                   v
               })
               .collect();

        debug!("RegionValueElements(num_universal_regions={:?})", num_universal_regions);
        debug!("RegionValueElements: statements_before_block={:#?}", statements_before_block);
        debug!("RegionValueElements: num_points={:#?}", num_points);

        Self {
            statements_before_block,
            num_universal_regions,
            num_points,
        }
    }

    /// Total number of element indices that exist.
    pub(super) fn num_elements(&self) -> usize {
        self.num_points + self.num_universal_regions
    }

    /// Converts an element of a region value into a `RegionElementIndex`.
    pub(super) fn index<T: ToElementIndex>(&self, elem: T) -> RegionElementIndex {
        elem.to_element_index(self)
    }

    /// Iterates over the `RegionElementIndex` for all points in the CFG.
    pub(super) fn all_point_indices<'a>(&'a self) -> impl Iterator<Item = RegionElementIndex> + 'a {
        (0..self.num_points).map(move |i| RegionElementIndex::new(i + self.num_universal_regions))
    }

    /// Iterates over the `RegionElementIndex` for all points in the CFG.
    pub(super) fn all_universal_region_indices(&self) -> impl Iterator<Item = RegionElementIndex> {
        (0..self.num_universal_regions).map(move |i| RegionElementIndex::new(i))
    }

    /// Converts a particular `RegionElementIndex` to the `RegionElement` it represents.
    pub(super) fn to_element(&self, i: RegionElementIndex) -> RegionElement {
        debug!("to_element(i={:?})", i);

        if let Some(r) = self.to_universal_region(i) {
            RegionElement::UniversalRegion(r)
        } else {
            let point_index = i.index() - self.num_universal_regions;

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
            let (block, &first_index) =
                self.statements_before_block
                    .iter_enumerated()
                    .filter(|(_, first_index)| **first_index <= point_index)
                    .last()
                    .unwrap();

            RegionElement::Location(Location {
                block,
                statement_index: point_index - first_index,
            })
        }
    }

    /// Converts a particular `RegionElementIndex` to a universal
    /// region, if that is what it represents. Returns `None`
    /// otherwise.
    pub(super) fn to_universal_region(&self, i: RegionElementIndex) -> Option<RegionVid> {
        if i.index() < self.num_universal_regions {
            Some(RegionVid::new(i.index()))
        } else {
            None
        }
    }
}

/// A newtype for the integers that represent one of the possible
/// elements in a region. These are the rows in the `BitMatrix` that
/// is used to store the values of all regions. They have the following
/// convention:
///
/// - The first N indices represent free regions (where N = universal_regions.len()).
/// - The remainder represent the points in the CFG (see `point_indices` map).
///
/// You can convert a `RegionElementIndex` into a `RegionElement`
/// using the `to_region_elem` method.
newtype_index!(RegionElementIndex { DEBUG_FORMAT = "RegionElementIndex({})" });

/// An individual element in a region value -- the value of a
/// particular region variable consists of a set of these elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum RegionElement {
    /// A point in the control-flow graph.
    Location(Location),

    /// An in-scope, universally quantified region (e.g., a liftime parameter).
    UniversalRegion(RegionVid),
}


pub(super) trait ToElementIndex {
    fn to_element_index(self, elements: &RegionValueElements) -> RegionElementIndex;
}

impl ToElementIndex for Location {
    fn to_element_index(self, elements: &RegionValueElements) -> RegionElementIndex {
        let Location { block, statement_index } = self;
        let start_index = elements.statements_before_block[block];
        RegionElementIndex::new(elements.num_universal_regions + start_index + statement_index)
    }
}

impl ToElementIndex for RegionVid {
    fn to_element_index(self, elements: &RegionValueElements) -> RegionElementIndex {
        assert!(self.index() < elements.num_universal_regions);
        RegionElementIndex::new(self.index())
    }
}

impl ToElementIndex for RegionElementIndex {
    fn to_element_index(self, _elements: &RegionValueElements) -> RegionElementIndex {
        self
    }
}

/// Stores the values for a set of regions. These are stored in a
/// compact `BitMatrix` representation, with one row per region
/// variable. The columns consist of either universal regions or
/// points in the CFG.
#[derive(Clone)]
pub(super) struct RegionValues {
    elements: Rc<RegionValueElements>,
    matrix: BitMatrix,
}

impl RegionValues {
    pub(super) fn new(elements: &Rc<RegionValueElements>, num_region_variables: usize) -> Self {
        assert!(
            elements.num_universal_regions <= num_region_variables,
            "universal regions are a subset of the region variables"
        );

        Self {
            elements: elements.clone(),
            matrix: BitMatrix::new(num_region_variables, elements.num_elements()),
        }
    }

    /// Adds the given element to the value for the given region. Returns true if
    /// the element is newly added (i.e., was not already present).
    pub(super) fn add<E: ToElementIndex>(&mut self, r: RegionVid, elem: E) -> bool {
        let i = self.elements.index(elem);
        if self.matrix.add(r.index(), i.index()) {
            debug!("add(r={:?}, i={:?})", r, self.elements.to_element(i));
            true
        } else {
            false
        }
    }

    /// Adds all the universal regions outlived by `from_region` to
    /// `to_region`.
    pub(super) fn add_universal_regions_outlived_by(
        &mut self,
        from_region: RegionVid,
        to_region: RegionVid,
    ) -> bool {
        // FIXME. We could optimize this by improving
        // `BitMatrix::merge` so it does not always merge an entire
        // row.
        debug!("add_universal_regions_outlived_by(from_region={:?}, to_region={:?})",
               from_region, to_region);
        let mut changed = false;
        for elem in self.elements.all_universal_region_indices() {
            if self.contains(from_region, elem) {
                changed |= self.add(to_region, elem);
            }
        }
        changed
    }

    /// True if the region `r` contains the given element.
    pub(super) fn contains<E: ToElementIndex>(&self, r: RegionVid, elem: E) -> bool {
        let i = self.elements.index(elem);
        self.matrix.contains(r.index(), i.index())
    }

    /// Iterate over the value of the region `r`, yielding up element
    /// indices. You may prefer `universal_regions_outlived_by` or
    /// `elements_contained_in`.
    pub(super) fn element_indices_contained_in<'a>(
        &'a self,
        r: RegionVid,
    ) -> impl Iterator<Item = RegionElementIndex> + 'a {
        self.matrix
            .iter(r.index())
            .map(move |i| RegionElementIndex::new(i))
    }

    /// Returns just the universal regions that are contained in a given region's value.
    pub(super) fn universal_regions_outlived_by<'a>(
        &'a self,
        r: RegionVid,
    ) -> impl Iterator<Item = RegionVid> + 'a {
        self.element_indices_contained_in(r)
            .map(move |i| self.elements.to_universal_region(i))
            .take_while(move |v| v.is_some()) // universal regions are a prefix
            .map(move |v| v.unwrap())
    }

    /// Returns all the elements contained in a given region's value.
    pub(super) fn elements_contained_in<'a>(
        &'a self,
        r: RegionVid,
    ) -> impl Iterator<Item = RegionElement> + 'a {
        self.element_indices_contained_in(r)
            .map(move |r| self.elements.to_element(r))
    }

    /// Returns a "pretty" string value of the region. Meant for debugging.
    pub(super) fn region_value_str(&self, r: RegionVid) -> String {
        let mut result = String::new();
        result.push_str("{");

        for (index, element) in self.elements_contained_in(r).enumerate() {
            if index > 0 {
                result.push_str(", ");
            }

            match element {
                RegionElement::Location(l) => {
                    result.push_str(&format!("{:?}", l));
                }

                RegionElement::UniversalRegion(fr) => {
                    result.push_str(&format!("{:?}", fr));
                }
            }
        }

        result.push_str("}");

        result
    }
}
