#![allow(unused)]

//! Implements a map from integer indices to data.
//! Rather than storing data for every index, internally, this maps entire ranges to the data.
//! To this end, the APIs all work on ranges, not on individual integers. Ranges are split as
//! necessary (e.g. when [0,5) is first associated with X, and then [1,2) is mutated).
//! Users must not depend on whether a range is coalesced or not, even though this is observable
//! via the iteration APIs.

use std::ops;
use std::num::NonZeroU64;

use rustc::ty::layout::Size;

// Representation: offset-length-data tuples, sorted by offset.
#[derive(Clone, Debug)]
struct Elem<T> {
    offset: u64,
    len: NonZeroU64,
    data: T,
}
// Length is always > 0.
#[derive(Clone, Debug)]
pub struct RangeMap<T> {
    v: Vec<Elem<T>>,
}

impl<T> Elem<T> {
    #[inline(always)]
    fn contains(&self, offset: u64) -> bool {
        offset >= self.offset && offset < self.offset + self.len.get()
    }
}

impl<T> RangeMap<T> {
    /// Create a new RangeMap for the given size, and with the given initial value used for
    /// the entire range.
    #[inline(always)]
    pub fn new(size: Size, init: T) -> RangeMap<T> {
        let size = size.bytes();
        let mut map = RangeMap { v: Vec::new() };
        if size > 0 {
            map.v.push(Elem {
                offset: 0,
                len: NonZeroU64::new(size).unwrap(),
                data: init
            });
        }
        map
    }

    /// Find the index containing the given offset.
    fn find_offset(&self, offset: u64) -> usize {
        debug_assert!(self.v.len() > 0);
        let mut left = 0usize; // inclusive
        let mut right = self.v.len(); // exclusive
        loop {
            let candidate = left.checked_add(right).unwrap() / 2;
            let elem = &self.v[candidate];
            if elem.offset > offset {
                // we are too far right (offset is further left)
                debug_assert!(candidate < right); // we are making progress
                right = candidate;
            } else if offset >= elem.offset + elem.len.get() {
                // we are too far left (offset is further right)
                debug_assert!(candidate >= left); // we are making progress
                left = candidate+1;
                debug_assert!(left < right, "find_offset: offset {} is out-of-bounds", offset);
            } else {
                // This is it!
                return candidate;
            }
        }
    }

    /// Provide read-only iteration over everything in the given range.  This does
    /// *not* split items if they overlap with the edges.  Do not use this to mutate
    /// through interior mutability.
    pub fn iter<'a>(&'a self, offset: Size, len: Size) -> impl Iterator<Item = &'a T> + 'a {
        let offset = offset.bytes();
        let len = len.bytes();
        // Compute a slice starting with the elements we care about
        let slice: &[Elem<T>] = if len == 0 {
                // We just need any empty iterator.  We don't even want to
                // yield the element that surrounds this position.
                &[]
            } else {
                let first = self.find_offset(offset);
                &self.v[first..]
            };
        let end = offset + len; // the first offset that is not included any more
        slice.iter()
            .take_while(move |elem| elem.offset < end)
            .map(|elem| &elem.data)
    }

    pub fn iter_mut_all<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> + 'a {
        self.v.iter_mut().map(|elem| &mut elem.data)
    }

    // Split the element situated at the given `index`, such that the 2nd one starts at offset `split_offset`.
    // Do nothing if the element already starts there.
    // Return whether a split was necessary.
    fn split_index(&mut self, index: usize, split_offset: u64) -> bool
    where
        T: Clone,
    {
        let elem = &mut self.v[index];
        let first_len = split_offset.checked_sub(elem.offset)
            .expect("The split_offset is before the element to be split");
        assert!(first_len <= elem.len.get(),
            "The split_offset is after the element to be split");
        if first_len == 0 || first_len == elem.len.get() {
            // Nothing to do
            return false;
        }

        // Now we really have to split.  Reduce length of first element.
        let second_len = elem.len.get() - first_len;
        elem.len = NonZeroU64::new(first_len).unwrap();
        // Copy the data, and insert 2nd element
        let second = Elem {
            offset: split_offset,
            len: NonZeroU64::new(second_len).unwrap(),
            data: elem.data.clone(),
        };
        self.v.insert(index+1, second);
        return true;
    }

    /// Provide mutable iteration over everything in the given range.  As a side-effect,
    /// this will split entries in the map that are only partially hit by the given range,
    /// to make sure that when they are mutated, the effect is constrained to the given range.
    pub fn iter_mut<'a>(
        &'a mut self,
        offset: Size,
        len: Size,
    ) -> impl Iterator<Item = &'a mut T> + 'a
    where
        T: Clone,
    {
        let offset = offset.bytes();
        let len = len.bytes();
        // Compute a slice containing exactly the elements we care about
        let slice: &mut [Elem<T>] = if len == 0 {
                // We just need any empty iterator.  We don't even want to
                // yield the element that surrounds this position, nor do
                // any splitting.
                &mut []
            } else {
                // Make sure we got a clear beginning
                let mut first = self.find_offset(offset);
                if self.split_index(first, offset) {
                    // The newly created 2nd element is ours
                    first += 1;
                }
                let first = first; // no more mutation
                // Find our end.  Linear scan, but that's okay because the iteration
                // is doing the same linear scan anyway -- no increase in complexity.
                let mut end = first; // the last element to be included
                loop {
                    let elem = &self.v[end];
                    if elem.offset+elem.len.get() < offset+len {
                        // We need to scan further.
                        end += 1;
                        debug_assert!(end < self.v.len(), "iter_mut: end-offset {} is out-of-bounds", offset+len);
                    } else {
                        // `elem` is the last included element.  Stop search.
                        break;
                    }
                }
                let end = end; // no more mutation
                // We need to split the end as well.  Even if this performs a
                // split, we don't have to adjust our index as we only care about
                // the first part of the split.
                self.split_index(end, offset+len);
                // Now we yield the slice. `end` is inclusive.
                &mut self.v[first..=end]
            };
        slice.iter_mut().map(|elem| &mut elem.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Query the map at every offset in the range and collect the results.
    fn to_vec<T: Copy>(map: &RangeMap<T>, offset: u64, len: u64) -> Vec<T> {
        (offset..offset + len)
            .into_iter()
            .map(|i| map
                .iter(Size::from_bytes(i), Size::from_bytes(1))
                .next()
                .map(|&t| t)
                .unwrap()
            )
            .collect()
    }

    #[test]
    fn basic_insert() {
        let mut map = RangeMap::<i32>::new(Size::from_bytes(20), -1);
        // Insert
        for x in map.iter_mut(Size::from_bytes(10), Size::from_bytes(1)) {
            *x = 42;
        }
        // Check
        assert_eq!(to_vec(&map, 10, 1), vec![42]);
        assert_eq!(map.v.len(), 3);

        // Insert with size 0
        for x in map.iter_mut(Size::from_bytes(10), Size::from_bytes(0)) {
            *x = 19;
        }
        for x in map.iter_mut(Size::from_bytes(11), Size::from_bytes(0)) {
            *x = 19;
        }
        assert_eq!(to_vec(&map, 10, 2), vec![42, -1]);
        assert_eq!(map.v.len(), 3);
    }

    #[test]
    fn gaps() {
        let mut map = RangeMap::<i32>::new(Size::from_bytes(20), -1);
        for x in map.iter_mut(Size::from_bytes(11), Size::from_bytes(1)) {
            *x = 42;
        }
        for x in map.iter_mut(Size::from_bytes(15), Size::from_bytes(1)) {
            *x = 43;
        }
        assert_eq!(map.v.len(), 5);
        assert_eq!(
            to_vec(&map, 10, 10),
            vec![-1, 42, -1, -1, -1, 43, -1, -1, -1, -1]
        );

        for x in map.iter_mut(Size::from_bytes(10), Size::from_bytes(10)) {
            if *x < 42 {
                *x = 23;
            }
        }
        assert_eq!(map.v.len(), 6);

        assert_eq!(
            to_vec(&map, 10, 10),
            vec![23, 42, 23, 23, 23, 43, 23, 23, 23, 23]
        );
        assert_eq!(to_vec(&map, 13, 5), vec![23, 23, 43, 23, 23]);

        for x in map.iter_mut(Size::from_bytes(15), Size::from_bytes(5)) {
            *x = 19;
        }
        assert_eq!(map.v.len(), 6);
        assert_eq!(map.iter(Size::from_bytes(19), Size::from_bytes(1))
            .map(|&t| t).collect::<Vec<_>>(), vec![19]);
    }
}
