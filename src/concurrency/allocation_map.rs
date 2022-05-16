//! Implements a map from allocation ranges to data.
//! This is somewhat similar to RangeMap, but the ranges
//! and data are discrete and non-splittable. An allocation in the
//! map will always have the same range until explicitly removed

use rustc_target::abi::Size;
use std::ops::{Index, IndexMut, Range};

use rustc_const_eval::interpret::AllocRange;

#[derive(Clone, Debug)]
struct Elem<T> {
    /// The range covered by this element; never empty.
    range: AllocRange,
    /// The data stored for this element.
    data: T,
}

/// Index of an allocation within the map
type Position = usize;

#[derive(Clone, Debug)]
pub struct AllocationMap<T> {
    v: Vec<Elem<T>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AccessType {
    /// The access perfectly overlaps (same offset and range) with the exsiting allocation
    PerfectlyOverlapping(Position),
    /// The access does not touch any exising allocation
    Empty(Position),
    /// The access overlaps with one or more existing allocations
    ImperfectlyOverlapping(Range<Position>),
}

impl<T> AllocationMap<T> {
    pub fn new() -> Self {
        Self { v: Vec::new() }
    }

    /// Finds the position of the allocation containing the given offset. If the offset is not
    /// in an existing allocation, then returns Err containing the position
    /// where such allocation should be inserted
    fn find_offset(&self, offset: Size) -> Result<Position, Position> {
        // We do a binary search.
        let mut left = 0usize; // inclusive
        let mut right = self.v.len(); // exclusive
        loop {
            if left == right {
                // No element contains the given offset. But the
                // index is where such element should be placed at.
                return Err(left);
            }
            let candidate = left.checked_add(right).unwrap() / 2;
            let elem = &self.v[candidate];
            if offset < elem.range.start {
                // We are too far right (offset is further left).
                debug_assert!(candidate < right); // we are making progress
                right = candidate;
            } else if offset >= elem.range.end() {
                // We are too far left (offset is further right).
                debug_assert!(candidate >= left); // we are making progress
                left = candidate + 1;
            } else {
                // This is it!
                return Ok(candidate);
            }
        }
    }

    /// Determines whether a given access on `range` overlaps with
    /// an existing allocation
    pub fn access_type(&self, range: AllocRange) -> AccessType {
        match self.find_offset(range.start) {
            Ok(index) => {
                // Start of the range belongs to an existing object, now let's check the overlapping situation
                let elem = &self.v[index];
                // FIXME: derive Eq for AllocRange in rustc
                if elem.range.start == range.start && elem.range.size == range.size {
                    // Happy case: perfectly overlapping access
                    AccessType::PerfectlyOverlapping(index)
                } else {
                    // FIXME: add a last() method to AllocRange that returns the last inclusive offset (end() is exclusive)
                    let end_index = match self.find_offset(range.end() - Size::from_bytes(1)) {
                        // If the end lands in an existing object, add one to get the exclusive index
                        Ok(inclusive) => inclusive + 1,
                        Err(exclusive) => exclusive,
                    };

                    AccessType::ImperfectlyOverlapping(index..end_index)
                }
            }
            Err(index) => {
                // Start of the range doesn't belong to an existing object
                match self.find_offset(range.end() - Size::from_bytes(1)) {
                    // Neither does the end
                    Err(end_index) =>
                        if index == end_index {
                            // There's nothing between the start and the end, so the range thing is empty
                            AccessType::Empty(index)
                        } else {
                            // Otherwise we have entirely covered an existing object
                            AccessType::ImperfectlyOverlapping(index..end_index)
                        },
                    // Otherwise at least part of it overlaps with something else
                    Ok(end_index) => AccessType::ImperfectlyOverlapping(index..end_index + 1),
                }
            }
        }
    }

    /// Inserts an object and its occupied range at given position
    pub fn insert(&mut self, index: Position, range: AllocRange, data: T) {
        self.v.insert(index, Elem { range, data });
        // If we aren't the first element, then our start must be greater than the preivous element's end
        if index > 0 {
            debug_assert!(self.v[index - 1].range.end() <= range.start);
        }
        // If we aren't the last element, then our end must be smaller than next element's start
        if index < self.v.len() - 1 {
            debug_assert!(range.end() <= self.v[index + 1].range.start);
        }
    }
}

impl<T> Index<Position> for AllocationMap<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.v[index].data
    }
}

impl<T> IndexMut<Position> for AllocationMap<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.v[index].data
    }
}

#[cfg(test)]
mod tests {
    use rustc_const_eval::interpret::alloc_range;

    use super::*;

    #[test]
    fn empty_map() {
        // FIXME: make Size::from_bytes const
        let four = Size::from_bytes(4);
        let map = AllocationMap::<()>::new();

        // Correctly tells where we should insert the first element (at index 0)
        assert_eq!(map.find_offset(Size::from_bytes(3)), Err(0));

        // Correctly tells the access type along with the supposed index
        assert_eq!(map.access_type(alloc_range(Size::ZERO, four)), AccessType::Empty(0));
    }

    #[test]
    #[should_panic]
    fn no_overlapping_inserts() {
        let four = Size::from_bytes(4);

        let mut map = AllocationMap::<&str>::new();

        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert(0, alloc_range(four, four), "#");
        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 ^ ^ ^ ^ 5 6 7 8 9 a b c d
        map.insert(0, alloc_range(Size::from_bytes(1), four), "@");
    }

    #[test]
    fn boundaries() {
        let four = Size::from_bytes(4);

        let mut map = AllocationMap::<&str>::new();

        // |#|#|#|#|_|_|...
        //  0 1 2 3 4 5
        map.insert(0, alloc_range(Size::ZERO, four), "#");
        // |#|#|#|#|_|_|...
        //  0 1 2 3 ^ 5
        assert_eq!(map.find_offset(four), Err(1));
        // |#|#|#|#|_|_|_|_|_|...
        //  0 1 2 3 ^ ^ ^ ^ 8
        assert_eq!(map.access_type(alloc_range(four, four)), AccessType::Empty(1));

        let eight = Size::from_bytes(8);
        // |#|#|#|#|_|_|_|_|@|@|@|@|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert(1, alloc_range(eight, four), "@");
        // |#|#|#|#|_|_|_|_|@|@|@|@|_|_|...
        //  0 1 2 3 4 5 6 ^ 8 9 a b c d
        assert_eq!(map.find_offset(Size::from_bytes(7)), Err(1));
        // |#|#|#|#|_|_|_|_|@|@|@|@|_|_|...
        //  0 1 2 3 ^ ^ ^ ^ 8 9 a b c d
        assert_eq!(map.access_type(alloc_range(four, four)), AccessType::Empty(1));
    }

    #[test]
    fn perfectly_overlapping() {
        let four = Size::from_bytes(4);

        let mut map = AllocationMap::<&str>::new();

        // |#|#|#|#|_|_|...
        //  0 1 2 3 4 5
        map.insert(0, alloc_range(Size::ZERO, four), "#");
        // |#|#|#|#|_|_|...
        //  ^ ^ ^ ^ 4 5
        assert_eq!(map.find_offset(Size::ZERO), Ok(0));
        assert_eq!(
            map.access_type(alloc_range(Size::ZERO, four)),
            AccessType::PerfectlyOverlapping(0)
        );

        // |#|#|#|#|@|@|@|@|_|...
        //  0 1 2 3 4 5 6 7 8
        map.insert(1, alloc_range(four, four), "@");
        // |#|#|#|#|@|@|@|@|_|...
        //  0 1 2 3 ^ ^ ^ ^ 8
        assert_eq!(map.find_offset(four), Ok(1));
        assert_eq!(map.access_type(alloc_range(four, four)), AccessType::PerfectlyOverlapping(1));
    }

    #[test]
    fn straddling() {
        let four = Size::from_bytes(4);

        let mut map = AllocationMap::<&str>::new();

        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert(0, alloc_range(four, four), "#");
        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 ^ ^ ^ ^ 6 7 8 9 a b c d
        assert_eq!(
            map.access_type(alloc_range(Size::from_bytes(2), four)),
            AccessType::ImperfectlyOverlapping(0..1)
        );
        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 2 3 4 5 ^ ^ ^ ^ a b c d
        assert_eq!(
            map.access_type(alloc_range(Size::from_bytes(6), four)),
            AccessType::ImperfectlyOverlapping(0..1)
        );
        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 ^ ^ ^ ^ ^ ^ ^ ^ a b c d
        assert_eq!(
            map.access_type(alloc_range(Size::from_bytes(2), Size::from_bytes(8))),
            AccessType::ImperfectlyOverlapping(0..1)
        );

        // |_|_|_|_|#|#|#|#|_|_|@|@|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert(1, alloc_range(Size::from_bytes(10), Size::from_bytes(2)), "@");
        // |_|_|_|_|#|#|#|#|_|_|@|@|_|_|...
        //  0 1 2 3 4 5 ^ ^ ^ ^ ^ ^ ^ ^
        assert_eq!(
            map.access_type(alloc_range(Size::from_bytes(6), Size::from_bytes(8))),
            AccessType::ImperfectlyOverlapping(0..2)
        );
    }
}
