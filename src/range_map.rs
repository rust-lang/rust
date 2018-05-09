//! Implements a map from integer indices to data.
//! Rather than storing data for every index, internally, this maps entire ranges to the data.
//! To this end, the APIs all work on ranges, not on individual integers. Ranges are split as
//! necessary (e.g. when [0,5) is first associated with X, and then [1,2) is mutated).
//! Users must not depend on whether a range is coalesced or not, even though this is observable
//! via the iteration APIs.
use std::collections::BTreeMap;
use std::ops;

#[derive(Clone, Debug)]
pub struct RangeMap<T> {
    map: BTreeMap<Range, T>,
}

// The derived `Ord` impl sorts first by the first field, then, if the fields are the same,
// by the second field.
// This is exactly what we need for our purposes, since a range query on a BTReeSet/BTreeMap will give us all
// `MemoryRange`s whose `start` is <= than the one we're looking for, but not > the end of the range we're checking.
// At the same time the `end` is irrelevant for the sorting and range searching, but used for the check.
// This kind of search breaks, if `end < start`, so don't do that!
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
struct Range {
    start: u64,
    end: u64, // Invariant: end > start
}

impl Range {
    fn range(offset: u64, len: u64) -> ops::Range<Range> {
        assert!(len > 0);
        // We select all elements that are within
        // the range given by the offset into the allocation and the length.
        // This is sound if all ranges that intersect with the argument range, are in the
        // resulting range of ranges.
        let left = Range {
            // lowest range to include `offset`
            start: 0,
            end: offset + 1,
        };
        let right = Range {
            // lowest (valid) range not to include `offset+len`
            start: offset + len,
            end: offset + len + 1,
        };
        left..right
    }

    /// Tests if all of [offset, offset+len) are contained in this range.
    fn overlaps(&self, offset: u64, len: u64) -> bool {
        assert!(len > 0);
        offset < self.end && offset + len >= self.start
    }
}

impl<T> RangeMap<T> {
    pub fn new() -> RangeMap<T> {
        RangeMap { map: BTreeMap::new() }
    }

    fn iter_with_range<'a>(
        &'a self,
        offset: u64,
        len: u64,
    ) -> impl Iterator<Item = (&'a Range, &'a T)> + 'a {
        assert!(len > 0);
        self.map.range(Range::range(offset, len)).filter_map(
            move |(range,
                   data)| {
                if range.overlaps(offset, len) {
                    Some((range, data))
                } else {
                    None
                }
            },
        )
    }

    pub fn iter<'a>(&'a self, offset: u64, len: u64) -> impl Iterator<Item = &'a T> + 'a {
        self.iter_with_range(offset, len).map(|(_, data)| data)
    }

    fn split_entry_at(&mut self, offset: u64)
    where
        T: Clone,
    {
        let range = match self.iter_with_range(offset, 1).next() {
            Some((&range, _)) => range,
            None => return,
        };
        assert!(
            range.start <= offset && range.end > offset,
            "We got a range that doesn't even contain what we asked for."
        );
        // There is an entry overlapping this position, see if we have to split it
        if range.start < offset {
            let data = self.map.remove(&range).unwrap();
            let old = self.map.insert(
                Range {
                    start: range.start,
                    end: offset,
                },
                data.clone(),
            );
            assert!(old.is_none());
            let old = self.map.insert(
                Range {
                    start: offset,
                    end: range.end,
                },
                data,
            );
            assert!(old.is_none());
        }
    }

    pub fn iter_mut_all<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.map.values_mut()
    }

    /// Provide mutable iteration over everything in the given range.  As a side-effect,
    /// this will split entries in the map that are only partially hit by the given range,
    /// to make sure that when they are mutated, the effect is constrained to the given range.
    pub fn iter_mut_with_gaps<'a>(
        &'a mut self,
        offset: u64,
        len: u64,
    ) -> impl Iterator<Item = &'a mut T> + 'a
    where
        T: Clone,
    {
        assert!(len > 0);
        // Preparation: Split first and last entry as needed.
        self.split_entry_at(offset);
        self.split_entry_at(offset + len);
        // Now we can provide a mutable iterator
        self.map.range_mut(Range::range(offset, len)).filter_map(
            move |(&range, data)| {
                if range.overlaps(offset, len) {
                    assert!(
                        offset <= range.start && offset + len >= range.end,
                        "The splitting went wrong"
                    );
                    Some(data)
                } else {
                    // Skip this one
                    None
                }
            },
        )
    }

    /// Provide a mutable iterator over everything in the given range, with the same side-effects as
    /// iter_mut_with_gaps.  Furthermore, if there are gaps between ranges, fill them with the given default.
    /// This is also how you insert.
    pub fn iter_mut<'a>(&'a mut self, offset: u64, len: u64) -> impl Iterator<Item = &'a mut T> + 'a
    where
        T: Clone + Default,
    {
        // Do a first iteration to collect the gaps
        let mut gaps = Vec::new();
        let mut last_end = offset;
        for (range, _) in self.iter_with_range(offset, len) {
            if last_end < range.start {
                gaps.push(Range {
                    start: last_end,
                    end: range.start,
                });
            }
            last_end = range.end;
        }
        if last_end < offset + len {
            gaps.push(Range {
                start: last_end,
                end: offset + len,
            });
        }

        // Add default for all gaps
        for gap in gaps {
            let old = self.map.insert(gap, Default::default());
            assert!(old.is_none());
        }

        // Now provide mutable iteration
        self.iter_mut_with_gaps(offset, len)
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let mut remove = Vec::new();
        for (range, data) in self.map.iter() {
            if !f(data) {
                remove.push(*range);
            }
        }

        for range in remove {
            self.map.remove(&range);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Query the map at every offset in the range and collect the results.
    fn to_vec<T: Copy>(map: &RangeMap<T>, offset: u64, len: u64) -> Vec<T> {
        (offset..offset + len)
            .into_iter()
            .map(|i| *map.iter(i, 1).next().unwrap())
            .collect()
    }

    #[test]
    fn basic_insert() {
        let mut map = RangeMap::<i32>::new();
        // Insert
        for x in map.iter_mut(10, 1) {
            *x = 42;
        }
        // Check
        assert_eq!(to_vec(&map, 10, 1), vec![42]);
    }

    #[test]
    fn gaps() {
        let mut map = RangeMap::<i32>::new();
        for x in map.iter_mut(11, 1) {
            *x = 42;
        }
        for x in map.iter_mut(15, 1) {
            *x = 42;
        }

        // Now request a range that needs three gaps filled
        for x in map.iter_mut(10, 10) {
            if *x != 42 {
                *x = 23;
            }
        }

        assert_eq!(
            to_vec(&map, 10, 10),
            vec![23, 42, 23, 23, 23, 42, 23, 23, 23, 23]
        );
        assert_eq!(to_vec(&map, 13, 5), vec![23, 23, 42, 23, 23]);
    }
}
