//! Implements a map from allocation ranges to data. This is somewhat similar to RangeMap, but the
//! ranges and data are discrete and non-splittable -- they represent distinct "objects". An
//! allocation in the map will always have the same range until explicitly removed

use std::ops::Range;

use rustc_abi::Size;
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
pub struct RangeObjectMap<T> {
    v: Vec<Elem<T>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AccessType {
    /// The access perfectly overlaps (same offset and range) with an existing object.
    PerfectlyOverlapping(Position),
    /// The access does not touch any existing object.
    Empty(Position),
    /// The access overlaps with one or more existing objects.
    ImperfectlyOverlapping(Range<Position>),
}

impl<T> RangeObjectMap<T> {
    pub fn new() -> Self {
        Self { v: Vec::new() }
    }

    /// Finds the position of the allocation containing the given offset. If the offset is not
    /// in an existing allocation, then returns Err containing the position
    /// where such allocation should be inserted
    fn find_offset(&self, offset: Size) -> Result<Position, Position> {
        self.v.binary_search_by(|elem| -> std::cmp::Ordering {
            if offset < elem.range.start {
                // We are too far right (offset is further left).
                // (`Greater` means that `elem` is greater than the desired target.)
                std::cmp::Ordering::Greater
            } else if offset >= elem.range.end() {
                // We are too far left (offset is further right).
                std::cmp::Ordering::Less
            } else {
                // This is it!
                std::cmp::Ordering::Equal
            }
        })
    }

    /// Determines whether a given access on `range` overlaps with
    /// an existing allocation
    pub fn access_type(&self, range: AllocRange) -> AccessType {
        match self.find_offset(range.start) {
            Ok(pos) => {
                // Start of the range belongs to an existing object, now let's check the overlapping situation
                let elem = &self.v[pos];
                // FIXME: derive Eq for AllocRange in rustc
                if elem.range.start == range.start && elem.range.size == range.size {
                    // Happy case: perfectly overlapping access
                    AccessType::PerfectlyOverlapping(pos)
                } else {
                    // FIXME: add a last() method to AllocRange that returns the last inclusive offset (end() is exclusive)
                    let end_pos = match self.find_offset(range.end() - Size::from_bytes(1)) {
                        // If the end lands in an existing object, add one to get the exclusive position
                        Ok(inclusive_pos) => inclusive_pos + 1,
                        Err(exclusive_pos) => exclusive_pos,
                    };

                    AccessType::ImperfectlyOverlapping(pos..end_pos)
                }
            }
            Err(pos) => {
                // Start of the range doesn't belong to an existing object
                match self.find_offset(range.end() - Size::from_bytes(1)) {
                    // Neither does the end
                    Err(end_pos) =>
                        if pos == end_pos {
                            // There's nothing between the start and the end, so the range thing is empty
                            AccessType::Empty(pos)
                        } else {
                            // Otherwise we have entirely covered an existing object
                            AccessType::ImperfectlyOverlapping(pos..end_pos)
                        },
                    // Otherwise at least part of it overlaps with something else
                    Ok(end_pos) => AccessType::ImperfectlyOverlapping(pos..end_pos + 1),
                }
            }
        }
    }

    /// Inserts an object and its occupied range at given position
    // The Position can be calculated from AllocRange, but the only user of AllocationMap
    // always calls access_type before calling insert/index/index_mut, and we don't
    // want to repeat the binary search on each time, so we ask the caller to supply Position
    pub fn insert_at_pos(&mut self, pos: Position, range: AllocRange, data: T) {
        assert!(range.size > Size::ZERO);

        self.v.insert(pos, Elem { range, data });

        // Validate position.
        // If we aren't the first element, then our start must be greater than the previous element's end
        if pos > 0 {
            assert!(self.v[pos - 1].range.end() <= range.start);
        }
        // If we aren't the last element, then our end must be smaller than next element's start
        if pos < self.v.len() - 1 {
            assert!(range.end() <= self.v[pos + 1].range.start);
        }
    }

    pub fn remove_pos_range(&mut self, pos_range: Range<Position>) {
        self.v.drain(pos_range);
    }

    pub fn remove_from_pos(&mut self, pos: Position) -> T {
        self.v.remove(pos).data
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.v.iter().map(|e| &e.data)
    }

    pub fn range(&self, range: AllocRange) -> impl Iterator<Item = &T> {
        let begin = self.find_offset(range.start).unwrap_or_else(|i| i);
        self.v[begin..]
            .iter()
            .take_while(move |elem| {
                debug_assert!(elem.range.end() > range.start);
                // `elem` overlaps `range` if it starts before the end of `range`.
                elem.range.start < range.end()
            })
            .map(|e| &e.data)
    }

    pub fn remove_range_if(&mut self, range: AllocRange, pred: impl Fn(&T) -> bool) {
        let mut idx = self.find_offset(range.start).unwrap_or_else(|i| i);
        loop {
            let Some(elem) = self.v.get(idx) else { break };
            // If we no longer overlap, we can end the loop.
            if elem.range.start >= range.end() {
                break;
            }
            // See if we should remove this element.
            if pred(&elem.data) {
                self.v.remove(idx);
                // `idx` now refers to the next element.
            } else {
                // If not, go look at next element.
                idx += 1;
            }
        }
    }

    /// Get the element that starts at the given offset.
    pub fn get_start(&self, offset: Size) -> Option<&T> {
        let idx = self.find_offset(offset).ok()?;
        let elem = &self.v[idx];
        (elem.range.start == offset).then_some(&elem.data)
    }

    /// Get the element that starts at the given offset.
    pub fn get_start_mut(&mut self, offset: Size) -> Option<&mut T> {
        let idx = self.find_offset(offset).ok()?;
        let elem = &mut self.v[idx];
        (elem.range.start == offset).then_some(&mut elem.data)
    }

    pub fn get_pos(&self, pos: Position) -> Option<&T> {
        self.v.get(pos).map(|e| &e.data)
    }

    pub fn get_mut_pos(&mut self, pos: Position) -> Option<&mut T> {
        self.v.get_mut(pos).map(|e| &mut e.data)
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
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
        let map = RangeObjectMap::<()>::new();

        // Correctly tells where we should insert the first element (at position 0)
        assert_eq!(map.find_offset(Size::from_bytes(3)), Err(0));

        // Correctly tells the access type along with the supposed position
        assert_eq!(map.access_type(alloc_range(Size::ZERO, four)), AccessType::Empty(0));
    }

    #[test]
    #[should_panic]
    fn no_overlapping_inserts() {
        let four = Size::from_bytes(4);

        let mut map = RangeObjectMap::<&str>::new();

        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert_at_pos(0, alloc_range(four, four), "#");
        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 ^ ^ ^ ^ 5 6 7 8 9 a b c d
        map.insert_at_pos(0, alloc_range(Size::from_bytes(1), four), "@");
    }

    #[test]
    fn boundaries() {
        let four = Size::from_bytes(4);

        let mut map = RangeObjectMap::<&str>::new();

        // |#|#|#|#|_|_|...
        //  0 1 2 3 4 5
        map.insert_at_pos(0, alloc_range(Size::ZERO, four), "#");
        // |#|#|#|#|_|_|...
        //  0 1 2 3 ^ 5
        assert_eq!(map.find_offset(four), Err(1));
        // |#|#|#|#|_|_|_|_|_|...
        //  0 1 2 3 ^ ^ ^ ^ 8
        assert_eq!(map.access_type(alloc_range(four, four)), AccessType::Empty(1));

        let eight = Size::from_bytes(8);
        // |#|#|#|#|_|_|_|_|@|@|@|@|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert_at_pos(1, alloc_range(eight, four), "@");
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

        let mut map = RangeObjectMap::<&str>::new();

        // |#|#|#|#|_|_|...
        //  0 1 2 3 4 5
        map.insert_at_pos(0, alloc_range(Size::ZERO, four), "#");
        // |#|#|#|#|_|_|...
        //  ^ ^ ^ ^ 4 5
        assert_eq!(map.find_offset(Size::ZERO), Ok(0));
        assert_eq!(
            map.access_type(alloc_range(Size::ZERO, four)),
            AccessType::PerfectlyOverlapping(0)
        );

        // |#|#|#|#|@|@|@|@|_|...
        //  0 1 2 3 4 5 6 7 8
        map.insert_at_pos(1, alloc_range(four, four), "@");
        // |#|#|#|#|@|@|@|@|_|...
        //  0 1 2 3 ^ ^ ^ ^ 8
        assert_eq!(map.find_offset(four), Ok(1));
        assert_eq!(map.access_type(alloc_range(four, four)), AccessType::PerfectlyOverlapping(1));
    }

    #[test]
    fn straddling() {
        let four = Size::from_bytes(4);

        let mut map = RangeObjectMap::<&str>::new();

        // |_|_|_|_|#|#|#|#|_|_|_|_|...
        //  0 1 2 3 4 5 6 7 8 9 a b c d
        map.insert_at_pos(0, alloc_range(four, four), "#");
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
        map.insert_at_pos(1, alloc_range(Size::from_bytes(10), Size::from_bytes(2)), "@");
        // |_|_|_|_|#|#|#|#|_|_|@|@|_|_|...
        //  0 1 2 3 4 5 ^ ^ ^ ^ ^ ^ ^ ^
        assert_eq!(
            map.access_type(alloc_range(Size::from_bytes(6), Size::from_bytes(8))),
            AccessType::ImperfectlyOverlapping(0..2)
        );
    }
}
