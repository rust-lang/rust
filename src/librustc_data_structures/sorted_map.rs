// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::convert::From;
use std::mem;
use std::ops::{RangeBounds, Bound, Index, IndexMut};

/// `SortedMap` is a data structure with similar characteristics as BTreeMap but
/// slightly different trade-offs: lookup, inseration, and removal are O(log(N))
/// and elements can be iterated in order cheaply.
///
/// `SortedMap` can be faster than a `BTreeMap` for small sizes (<50) since it
/// stores data in a more compact way. It also supports accessing contiguous
/// ranges of elements as a slice, and slices of already sorted elements can be
/// inserted efficiently.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug, RustcEncodable,
         RustcDecodable)]
pub struct SortedMap<K: Ord, V> {
    data: Vec<(K,V)>
}

impl<K: Ord, V> SortedMap<K, V> {

    #[inline]
    pub fn new() -> SortedMap<K, V> {
        SortedMap {
            data: vec![]
        }
    }

    /// Construct a `SortedMap` from a presorted set of elements. This is faster
    /// than creating an empty map and then inserting the elements individually.
    ///
    /// It is up to the caller to make sure that the elements are sorted by key
    /// and that there are no duplicates.
    #[inline]
    pub fn from_presorted_elements(elements: Vec<(K, V)>) -> SortedMap<K, V>
    {
        debug_assert!(elements.windows(2).all(|w| w[0].0 < w[1].0));

        SortedMap {
            data: elements
        }
    }

    #[inline]
    pub fn insert(&mut self, key: K, mut value: V) -> Option<V> {
        match self.lookup_index_for(&key) {
            Ok(index) => {
                let slot = unsafe {
                    self.data.get_unchecked_mut(index)
                };
                mem::swap(&mut slot.1, &mut value);
                Some(value)
            }
            Err(index) => {
                self.data.insert(index, (key, value));
                None
            }
        }
    }

    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.lookup_index_for(key) {
            Ok(index) => {
                Some(self.data.remove(index).1)
            }
            Err(_) => {
                None
            }
        }
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        match self.lookup_index_for(key) {
            Ok(index) => {
                unsafe {
                    Some(&self.data.get_unchecked(index).1)
                }
            }
            Err(_) => {
                None
            }
        }
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.lookup_index_for(key) {
            Ok(index) => {
                unsafe {
                    Some(&mut self.data.get_unchecked_mut(index).1)
                }
            }
            Err(_) => {
                None
            }
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Iterate over elements, sorted by key
    #[inline]
    pub fn iter(&self) -> ::std::slice::Iter<(K, V)> {
        self.data.iter()
    }

    /// Iterate over the keys, sorted
    #[inline]
    pub fn keys(&self) -> impl Iterator<Item=&K> + ExactSizeIterator {
        self.data.iter().map(|&(ref k, _)| k)
    }

    /// Iterate over values, sorted by key
    #[inline]
    pub fn values(&self) -> impl Iterator<Item=&V> + ExactSizeIterator {
        self.data.iter().map(|&(_, ref v)| v)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn range<R>(&self, range: R) -> &[(K, V)]
        where R: RangeBounds<K>
    {
        let (start, end) = self.range_slice_indices(range);
        (&self.data[start .. end])
    }

    #[inline]
    pub fn remove_range<R>(&mut self, range: R)
        where R: RangeBounds<K>
    {
        let (start, end) = self.range_slice_indices(range);
        self.data.splice(start .. end, ::std::iter::empty());
    }

    /// Mutate all keys with the given function `f`. This mutation must not
    /// change the sort-order of keys.
    #[inline]
    pub fn offset_keys<F>(&mut self, f: F)
        where F: Fn(&mut K)
    {
        self.data.iter_mut().map(|&mut (ref mut k, _)| k).for_each(f);
    }

    /// Inserts a presorted range of elements into the map. If the range can be
    /// inserted as a whole in between to existing elements of the map, this
    /// will be faster than inserting the elements individually.
    ///
    /// It is up to the caller to make sure that the elements are sorted by key
    /// and that there are no duplicates.
    #[inline]
    pub fn insert_presorted(&mut self, mut elements: Vec<(K, V)>) {
        if elements.is_empty() {
            return
        }

        debug_assert!(elements.windows(2).all(|w| w[0].0 < w[1].0));

        let start_index = self.lookup_index_for(&elements[0].0);

        let drain = match start_index {
            Ok(index) => {
                let mut drain = elements.drain(..);
                self.data[index] = drain.next().unwrap();
                drain
            }
            Err(index) => {
                if index == self.data.len() ||
                   elements.last().unwrap().0 < self.data[index].0 {
                    // We can copy the whole range without having to mix with
                    // existing elements.
                    self.data.splice(index .. index, elements.drain(..));
                    return
                }

                let mut drain = elements.drain(..);
                self.data.insert(index, drain.next().unwrap());
                drain
            }
        };

        // Insert the rest
        for (k, v) in drain {
            self.insert(k, v);
        }
    }

    /// Looks up the key in `self.data` via `slice::binary_search()`.
    #[inline(always)]
    fn lookup_index_for(&self, key: &K) -> Result<usize, usize> {
        self.data.binary_search_by(|&(ref x, _)| x.cmp(key))
    }

    #[inline]
    fn range_slice_indices<R>(&self, range: R) -> (usize, usize)
        where R: RangeBounds<K>
    {
        let start = match range.start_bound() {
            Bound::Included(ref k) => {
                match self.lookup_index_for(k) {
                    Ok(index) | Err(index) => index
                }
            }
            Bound::Excluded(ref k) => {
                match self.lookup_index_for(k) {
                    Ok(index) => index + 1,
                    Err(index) => index,
                }
            }
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(ref k) => {
                match self.lookup_index_for(k) {
                    Ok(index) => index + 1,
                    Err(index) => index,
                }
            }
            Bound::Excluded(ref k) => {
                match self.lookup_index_for(k) {
                    Ok(index) | Err(index) => index,
                }
            }
            Bound::Unbounded => self.data.len(),
        };

        (start, end)
    }
}

impl<K: Ord, V> IntoIterator for SortedMap<K, V> {
    type Item = (K, V);
    type IntoIter = ::std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<K: Ord, V, Q: Borrow<K>> Index<Q> for SortedMap<K, V> {
    type Output = V;
    fn index(&self, index: Q) -> &Self::Output {
        let k: &K = index.borrow();
        self.get(k).unwrap()
    }
}

impl<K: Ord, V, Q: Borrow<K>> IndexMut<Q> for SortedMap<K, V> {
    fn index_mut(&mut self, index: Q) -> &mut Self::Output {
        let k: &K = index.borrow();
        self.get_mut(k).unwrap()
    }
}

impl<K: Ord, V, I: Iterator<Item=(K, V)>> From<I> for SortedMap<K, V> {
    fn from(data: I) -> Self {
        let mut data: Vec<(K, V)> = data.collect();
        data.sort_unstable_by(|&(ref k1, _), &(ref k2, _)| k1.cmp(k2));
        data.dedup_by(|&mut (ref k1, _), &mut (ref k2, _)| {
            k1.cmp(k2) == Ordering::Equal
        });
        SortedMap {
            data
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SortedMap;

    #[test]
    fn test_insert_and_iter() {
        let mut map = SortedMap::new();
        let mut expected = Vec::new();

        for x in 0 .. 100 {
            assert_eq!(map.iter().cloned().collect::<Vec<_>>(), expected);

            let x = 1000 - x * 2;
            map.insert(x, x);
            expected.insert(0, (x, x));
        }
    }

    #[test]
    fn test_get_and_index() {
        let mut map = SortedMap::new();
        let mut expected = Vec::new();

        for x in 0 .. 100 {
            let x = 1000 - x;
            if x & 1 == 0 {
                map.insert(x, x);
            }
            expected.push(x);
        }

        for mut x in expected {
            if x & 1 == 0 {
                assert_eq!(map.get(&x), Some(&x));
                assert_eq!(map.get_mut(&x), Some(&mut x));
                assert_eq!(map[&x], x);
                assert_eq!(&mut map[&x], &mut x);
            } else {
                assert_eq!(map.get(&x), None);
                assert_eq!(map.get_mut(&x), None);
            }
        }
    }

    #[test]
    fn test_range() {
        let mut map = SortedMap::new();
        map.insert(1, 1);
        map.insert(3, 3);
        map.insert(6, 6);
        map.insert(9, 9);

        let keys = |s: &[(_, _)]| {
            s.into_iter().map(|e| e.0).collect::<Vec<u32>>()
        };

        for start in 0 .. 11 {
            for end in 0 .. 11 {
                if end < start {
                    continue
                }

                let mut expected = vec![1, 3, 6, 9];
                expected.retain(|&x| x >= start && x < end);

                assert_eq!(keys(map.range(start..end)), expected, "range = {}..{}", start, end);
            }
        }
    }


    #[test]
    fn test_offset_keys() {
        let mut map = SortedMap::new();
        map.insert(1, 1);
        map.insert(3, 3);
        map.insert(6, 6);

        map.offset_keys(|k| *k += 1);

        let mut expected = SortedMap::new();
        expected.insert(2, 1);
        expected.insert(4, 3);
        expected.insert(7, 6);

        assert_eq!(map, expected);
    }

    fn keys(s: SortedMap<u32, u32>) -> Vec<u32> {
        s.into_iter().map(|(k, _)| k).collect::<Vec<u32>>()
    }

    fn elements(s: SortedMap<u32, u32>) -> Vec<(u32, u32)> {
        s.into_iter().collect::<Vec<(u32, u32)>>()
    }

    #[test]
    fn test_remove_range() {
        let mut map = SortedMap::new();
        map.insert(1, 1);
        map.insert(3, 3);
        map.insert(6, 6);
        map.insert(9, 9);

        for start in 0 .. 11 {
            for end in 0 .. 11 {
                if end < start {
                    continue
                }

                let mut expected = vec![1, 3, 6, 9];
                expected.retain(|&x| x < start || x >= end);

                let mut map = map.clone();
                map.remove_range(start .. end);

                assert_eq!(keys(map), expected, "range = {}..{}", start, end);
            }
        }
    }

    #[test]
    fn test_remove() {
        let mut map = SortedMap::new();
        let mut expected = Vec::new();

        for x in 0..10 {
            map.insert(x, x);
            expected.push((x, x));
        }

        for x in 0 .. 10 {
            let mut map = map.clone();
            let mut expected = expected.clone();

            assert_eq!(map.remove(&x), Some(x));
            expected.remove(x as usize);

            assert_eq!(map.iter().cloned().collect::<Vec<_>>(), expected);
        }
    }

    #[test]
    fn test_insert_presorted_non_overlapping() {
        let mut map = SortedMap::new();
        map.insert(2, 0);
        map.insert(8, 0);

        map.insert_presorted(vec![(3, 0), (7, 0)]);

        let expected = vec![2, 3, 7, 8];
        assert_eq!(keys(map), expected);
    }

    #[test]
    fn test_insert_presorted_first_elem_equal() {
        let mut map = SortedMap::new();
        map.insert(2, 2);
        map.insert(8, 8);

        map.insert_presorted(vec![(2, 0), (7, 7)]);

        let expected = vec![(2, 0), (7, 7), (8, 8)];
        assert_eq!(elements(map), expected);
    }

    #[test]
    fn test_insert_presorted_last_elem_equal() {
        let mut map = SortedMap::new();
        map.insert(2, 2);
        map.insert(8, 8);

        map.insert_presorted(vec![(3, 3), (8, 0)]);

        let expected = vec![(2, 2), (3, 3), (8, 0)];
        assert_eq!(elements(map), expected);
    }

    #[test]
    fn test_insert_presorted_shuffle() {
        let mut map = SortedMap::new();
        map.insert(2, 2);
        map.insert(7, 7);

        map.insert_presorted(vec![(1, 1), (3, 3), (8, 8)]);

        let expected = vec![(1, 1), (2, 2), (3, 3), (7, 7), (8, 8)];
        assert_eq!(elements(map), expected);
    }

    #[test]
    fn test_insert_presorted_at_end() {
        let mut map = SortedMap::new();
        map.insert(1, 1);
        map.insert(2, 2);

        map.insert_presorted(vec![(3, 3), (8, 8)]);

        let expected = vec![(1, 1), (2, 2), (3, 3), (8, 8)];
        assert_eq!(elements(map), expected);
    }
}
