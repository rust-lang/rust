// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use collections::HashMap;
use collections::hashmap::{Entries, Keys, MoveEntries, MutEntries, Values};
use collections::PriorityQueue;
use core::clone::Clone;
use core::cmp::{max, min};
use core::num::Saturating;
use std::default::Default;
use std::fmt;
use std::fmt::Show;
use std::hash::Hash;
use std::iter::Repeat;

/// A Rust port of Pythons collections.Counter().
///
/// Struct for counting hashable items, Also known as bag or multiset. Stores
/// items and their counts as (key,value) pairs. Unlike Pythons Counter, it
/// only allows non-negative integers as counts (this is subject to change).

/// A wrapper around a (k, v) pair. This is needed since a (K, uint) tuple can't
/// be put in a PriorityQueue without K implementing TotalOrd.
#[deriving(TotalEq, PartialEq)]
struct Counted<K> {
    k: K,
    v: uint
}

impl<K: Hash + TotalEq> PartialOrd for Counted<K> {
    fn lt(&self, other: &Counted<K>) -> bool {
        self.v < other.v
    }
}

impl<K: Hash + TotalEq> TotalOrd for Counted<K> {
    fn cmp(&self, other: &Counted<K>) -> Ordering {
        self.v.cmp(&other.v)
    }
}

#[deriving(Clone)]
pub struct CountedSet<K> {
    data: HashMap<K, uint>
}

impl<K: Hash + TotalEq> CountedSet<K> {

    /// Returns the `n` most common elements and their counts in descending
    /// order. The time complexity is O(m + n log m), where `m` is the number
    /// of elements in the counter.
    pub fn most_common<'a>(&'a self, n: uint) -> Vec<(&'a K, uint)> {
        let n = if n <= self.data.len() { n } else { self.data.len() };
        let counted: Vec<Counted<&'a K>> = self.data
                                           .iter()
                                           .map(|(kk, vv)| Counted {k: kk, v: *vv})
                                           .collect();
        let mut pq = PriorityQueue::from_vec(counted);
        let mut n_most_common = Vec::with_capacity(n);

        for _ in range(0u, n) {
            let Counted { k, v } = pq.pop().unwrap();
            n_most_common.push((k, v));
        }
        n_most_common
    }

    /// Returns a vector with containing each element the same number of times
    /// as its count.
    // FIXME(schmee): Return FlatMap
    pub fn elements<'a>(&'a self) -> Vec<&'a K> {
        self.data.iter().flat_map(|(k,v)| Repeat::new(k).take(*v)).collect()
    }

    /// Returns the count of an element in the counter. Missing elements return 0.
    pub fn get(&self, k: &K) -> uint {
        *self.data.find(k).unwrap_or(&0u)
    }

    /// Return a new counter.
    pub fn new() -> CountedSet<K> {
        CountedSet{data: HashMap::new()}
    }

    // =============== HashMap aliases ===============

    pub fn keys<'a>(&'a self) -> Keys<'a, K, uint> {
        self.data.keys()
    }

    pub fn values<'a>(&'a self) -> Values<'a, K, uint> {
        self.data.values()
    }

    pub fn iter<'a>(&'a self) -> Entries<'a, K, uint> {
        self.data.iter()
    }

    pub fn mut_iter<'a>(&'a mut self) -> MutEntries<'a, K, uint> {
        self.data.mut_iter()
    }

    pub fn move_iter(self) -> MoveEntries<K, uint> {
        self.data.move_iter()
    }
}

impl<K: Clone + Hash + TotalEq> CountedSet<K> {
    /// Returns a new counter containing the either the minimum or maximum value of either counter.
    fn get_min_or_max(&self, other: &CountedSet<K>,
                      min_or_max: |uint, uint| -> uint) -> CountedSet<K> {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            result.data.insert_or_update_with(
                (*k).clone(), *v, |_, old| *old = min_or_max(*v, *old));
        }
        result
    }
}


impl<K: Clone + Hash + TotalEq> Add<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
    /// Adds counts from two counters.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use collections::CountedSet;
    ///
    /// let strings = ["a", "a", "a", "b", "b", "c"];
    /// let more_strings = ["a", "b", "c"];
    ///
    /// // CountedSet({a: 3, c: 1, b: 2})
    /// let count_strings: CountedSet<_> = strings.iter().map(|&x|x).collect();
    /// // CountedSet({a: 1, b: 1, c: 1})
    /// let count_more_strings: CountedSet<_> = more_strings.iter().map(|&x|x).collect();
    /// // CountedSet({a: 4, b: 3, c: 2})
    /// let sum = count_strings + count_more_strings;
    /// ```
    fn add(&self, other: &CountedSet<K>) -> CountedSet<K> {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            result.data.insert_or_update_with((*k).clone(), *v, |_, old| *old += *v);
        }
        result
    }
}

impl<K: Clone + Hash + TotalEq> Sub<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
    /// Subtracts counts from two counters. If the count for an item
    /// would be negative, it is instead counted as 0.
    fn sub(&self, other: &CountedSet<K>) -> CountedSet<K> {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            let value = result.data.find_copy(k);
            value.map_or(true, |old| result.data.insert((*k).clone(), old.saturating_sub(*v)));
        }
        result
    }
}

impl<K: Clone + Hash + TotalEq> BitAnd<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
    /// Returns a new counter containing the minimum value of either counter.
    fn bitand(&self, other: &CountedSet<K>) -> CountedSet<K> {
        self.get_min_or_max(other, min)
    }
}

impl<K: Clone + Hash + TotalEq> BitOr<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
    /// Returns a new counter containing the maximum value of either counter.
    fn bitor(&self, other: &CountedSet<K>) -> CountedSet<K> {
        self.get_min_or_max(other, max)
    }
}

impl<K: Hash + TotalEq> Default for CountedSet<K> {
    fn default() -> CountedSet<K> { CountedSet::new() }
}

impl<K: Hash + TotalEq> Map<K, uint> for CountedSet<K> {
    fn find<'a>(&'a self, k: &K) -> Option<&'a uint> {
        self.data.find(k)
    }

    fn contains_key(&self, k: &K) -> bool {
        self.data.contains_key(k)
    }
}

impl<K: Hash + TotalEq> MutableMap<K, uint> for CountedSet<K> {
    fn find_mut<'a>(&'a mut self, k: &K) -> Option<&'a mut uint> {
        self.data.find_mut(k)
    }

    fn swap(&mut self, k: K, v: uint) -> Option<uint> {
        self.data.swap(k ,v)
    }

    fn pop(&mut self, k: &K) -> Option<uint> {
        self.data.pop(k)
    }

    fn insert(&mut self, k: K, v: uint) -> bool {
        self.data.insert(k, v)
    }

    fn remove(&mut self, k: &K) -> bool {
        self.data.remove(k)
    }
}

impl<K: Hash + TotalEq> Mutable for CountedSet<K> {
    fn clear(&mut self) { self.data.clear() }
}

impl<K: Hash + TotalEq> Container for CountedSet<K> {
    fn len(&self) -> uint { self.data.len() }
}

impl<K: Hash + TotalEq> PartialEq for CountedSet<K> {
    fn eq(&self, other: &CountedSet<K>) -> bool {
        self.data == other.data
    }
}

impl<K: Hash + TotalEq> Extendable<K> for CountedSet<K> {
    fn extend<I: Iterator<K>>(&mut self, mut iter: I) {
        for item in iter {
            self.data.insert_or_update_with(item, 1u, |_, v| *v += 1);
        }
    }
}

impl<K: Hash + TotalEq> FromIterator<K> for CountedSet<K> {
    fn from_iter<I: Iterator<K>>(iter: I) -> CountedSet<K> {
        let mut counts: CountedSet<K> = CountedSet::new();
        counts.extend(iter);
        counts
    }
}

impl<K: Hash + TotalEq + Show> Show for CountedSet<K> {
    #[allow(unused_must_use)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, r"CountedSet("));
        self.data.fmt(f);
        write!(f, r")")
    }
}

mod test {

    use super::CountedSet;
    use collections::HashMap;

    #[test]
    fn test_add() {
        let strings = ["a", "a", "a", "b", "b", "c"];
        let more_strings = ["a", "b", "c"];
        let count: CountedSet<_> = strings.iter().map(|&x|x).collect();
        let other: CountedSet<_> = more_strings.iter().map(|&x|x).collect();
        let answer = [("a", 4u), ("b", 3), ("c", 2)];

        let add = count + other;
        let add2 = other + count;
        assert_eq!(add, add2);
        let a = add.data;
        let b = answer.iter().map(|&x|x).collect::<HashMap<&'static str, uint>>();
        assert_eq!(a, b);
    }

    #[test]
    fn test_subtract() {
        let strings = ["a", "a", "a", "b", "b", "c"];
        let more_strings = ["a", "b", "c", "c", "c"];
        let count: CountedSet<_> = strings.iter().map(|&x|x).collect();
        let other: CountedSet<_> = more_strings.iter().map(|&x|x).collect();
        let answer = [("a", 2u), ("b", 1), ("c", 0)];

        let sub = count - other;
        let a = sub.data;
        let b = answer.iter().map(|&x|x).collect::<HashMap<&'static str, uint>>();
        assert_eq!(a, b);
    }

    #[test]
    fn test_union() {
        let numbers = [1u, 1, 1, 2, 2, 3, 1];
        let more_numbers = [1u, 1, 2];
        let count: CountedSet<_> = numbers.iter().map(|&x|x).collect();
        let other: CountedSet<_> = more_numbers.iter().map(|&x|x).collect();
        let answer = [(1u, 4u), (2, 2), (3, 1)];

        let union = count | other;
        let union2 = other | count;
        assert_eq!(union, union2);
        let a = union.data;
        let b = answer.iter().map(|&x|x).collect::<HashMap<uint, uint>>();
        assert_eq!(a, b)
    }

    #[test]
    fn test_intersection() {
        let numbers = [1u, 1, 1, 2, 2, 3, 1];
        let more_numbers = [1u, 1, 2];
        let count: CountedSet<_> = numbers.iter().map(|&x|x).collect();
        let other: CountedSet<_> = more_numbers.iter().map(|&x|x).collect();
        let answer = [(1u, 2u), (2, 1), (3, 1)];

        let inter = count & other;
        let inter2 = other & count;
        assert_eq!(inter, inter2);
        let a = inter.data;
        let b = answer.iter().map(|&x|x).collect::<HashMap<uint, uint>>();
        assert_eq!(a, b)
    }

    #[test]
    fn test_elements() {
        let strings = ["a", "a", "a", "b", "b", "c"];
        let count: CountedSet<&'static str> = strings.iter().map(|&x|x).collect();
        let mut elems: Vec<_> = count.elements().move_iter().map(|&x|x).collect();
        elems.sort();
        assert!(elems.as_slice() == strings);
    }

    #[test]
    fn test_most_common() {
        let strings = ["a", "b", "a", "b", "b", "c"];
        let count: CountedSet<&'static str> = strings.iter().map(|&x|x).collect();

        let v: Vec<_> = count.most_common(2);
        let answer = [(&"b", 3u), (&"a", 2)];
        assert!(answer == v.as_slice())

            let v: Vec<_> = count.most_common(6);
        let answer = [(&"b", 3u), (&"a", 2), (&"c", 1)];
        assert!(answer == v.as_slice())
    }

    #[test]
    fn test_get() {
        let strings = ["a", "b", "a", "b", "b", "c"];
        let count: CountedSet<_> = strings.iter().map(|&x|x).collect();
        assert_eq!(count.get(&"a"), 2)
            assert_eq!(count.get(&"b"), 3)
            assert_eq!(count.get(&"c"), 1)
            assert_eq!(count.get(&"missing"), 0)
    }

    #[test]
    fn test_extend() {
        let strings = ["red", "green", "blue"];
        let more_strings = ["blue", "blue", "blue"];
        let answer = [("blue", 4u), ("red", 1), ("green", 1)];

        let mut count: CountedSet<_> = CountedSet::new();
        count.extend(strings.iter().map(|&k|k));
        count.extend(more_strings.iter().map(|&k|k));
        let a = count.data;
        let b: HashMap<&'static str, uint> = answer.iter().map(|&x|x).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_from_iter() {
        let strings = ["blue", "red", "red", "blue", "green", "green", "red"];
        let answer = [("red", 3u), ("blue", 2), ("green", 2)];

        let count: CountedSet<_> = strings.iter().map(|&x|x).collect();
        let a = count.data;
        let b = answer.iter().map(|&x|x).collect::<HashMap<&'static str, uint>>();
        assert_eq!(a, b);
    }
}
