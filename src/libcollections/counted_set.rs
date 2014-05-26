// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate collections;
extern crate core;

use collections::HashMap;
use collections::hashmap::{Entries, Keys, MoveEntries, Values};
use core::clone::Clone;
use core::cmp::{max, min};
use core::num::Saturating;
use std::default::Default;
use std::fmt;
use std::fmt::Show;
use std::hash::Hash;
use std::iter::Repeat;

static MISSING: uint = 0;

/// A Rust port of Pythons collections.Counter().
///
/// Struct for counting hashable items, Also known as bag or multiset. Stores
/// items and their counts as (key,value) pairs. Unlike Pythons Counter, it
/// only allows non-negative integers as counts (this is subject to change).

#[deriving(Clone)]
pub struct CountedSet<K> {
    data: HashMap<K, uint>
}

impl<K: Clone + Hash + TotalEq> CountedSet<K> {
    /// Adds counts from two counters.
    pub fn add(&self, other: &CountedSet<K>) -> CountedSet<K> {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            result.data.insert_or_update_with((*k).clone(), *v, |_, old| *old += *v);
        }
        result
    }

    /// Subtracts counts from two counters. If the count for an item
    /// would be negative, it is instead counted as 0.
    pub fn subtract(&self, other: &CountedSet<K>) -> CountedSet<K> {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            let value = result.data.find_copy(k);
            value.map_or(true, |old| result.data.insert((*k).clone(), old.saturating_sub(*v)));
        }
        result
    }

    /// Returns a new counter containing the maximum value of either counter.
    pub fn union(&self, other: &CountedSet<K>) -> CountedSet<K> {
        self.get_min_or_max(other, max)
    }

    /// Returns a new counter containing the minimum value of either counter.
    pub fn intersection(&self, other: &CountedSet<K>) -> CountedSet<K> {
        self.get_min_or_max(other, min)
    }

    fn get_min_or_max(&self, other: &CountedSet<K>,
                      min_or_max: |uint, uint| -> uint) -> CountedSet<K> {
        let mut result = self.clone();
        for (k, v) in other.iter() {
            result.data.insert_or_update_with(
                (*k).clone(), *v, |_, old| *old = min_or_max(*v, *old));
        }
        result
    }

    /// Returns the `n` most common elements and their counts in descending
    /// order.
    pub fn most_common(&self, n: uint) -> Vec<(K, uint)> {
        let data = self.data.clone();
        let mut n_most_common = data.move_iter().collect::<Vec<(K, uint)>>();
        n_most_common.sort_by(|&(_, a), &(_, b)| b.cmp(&a));
        n_most_common.truncate(n);
        n_most_common
    }

    /// Returns a vector with containing each element the same number of times
    /// as its count.
    // FIXME(schmee): Return FlatMap
    pub fn elements(&self) -> Vec<K> {
        let data = self.data.clone();
        data.move_iter().flat_map(|(k,v)| Repeat::new(k).take(v)).collect()
    }

    /// Returns the count of an element in the counter. Missing elements return 0.
    pub fn get<'a>(&'a self, k: &K) -> &'a uint {
        match self.data.find(k) {
            Some(v) => v,
            None => &'static MISSING
        }
    }

    /// Return a new counter.
    pub fn new() -> CountedSet<K> {
        CountedSet{data: HashMap::new()}
    }

    // =============== HashMap aliases ===============

    pub fn get_mut<'a>(&'a mut self, k: &K) -> &'a mut uint {
        self.data.get_mut(k)
    }

    pub fn find_copy(&self, k: &K) -> Option<uint> {
        self.data.find_copy(k)
    }

    pub fn get_copy(&self, k: &K) -> uint {
        self.data.get_copy(k)
    }

    pub fn keys<'a>(&'a self) -> Keys<'a, K, uint> {
        self.data.keys()
    }

    pub fn values<'a>(&'a self) -> Values<'a, K, uint> {
        self.data.values()
    }

    pub fn iter<'a>(&'a self) -> Entries<'a, K, uint> {
        self.data.iter()
    }

    pub fn move_iter(self) -> MoveEntries<K, uint> {
        self.data.move_iter()
    }
}

impl<K: Clone + Hash + TotalEq> Add<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
        fn add(&self, rhs: &CountedSet<K>) -> CountedSet<K> {
            self.add(rhs)
        }
}

impl<K: Clone + Hash + TotalEq> Sub<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
        fn sub(&self, rhs: &CountedSet<K>) -> CountedSet<K> {
            self.subtract(rhs)
        }
}

impl<K: Clone + Hash + TotalEq> BitAnd<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
        fn bitand(&self, rhs: &CountedSet<K>) -> CountedSet<K> {
            self.intersection(rhs)
        }
}

impl<K: Clone + Hash + TotalEq> BitOr<CountedSet<K>, CountedSet<K>> for CountedSet<K> {
        fn bitor(&self, rhs: &CountedSet<K>) -> CountedSet<K> {
            self.union(rhs)
        }
}

impl<K: Clone + Hash + TotalEq> Default for CountedSet<K> {
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

impl<K: Hash + TotalEq> Eq for CountedSet<K> {
    fn eq(&self, other: &CountedSet<K>) -> bool {
        self.data == other.data
    }
}

impl<K: Clone + Hash + TotalEq> Extendable<K> for CountedSet<K> {
    fn extend<I: Iterator<K>>(&mut self, mut iter: I) {
        for item in iter {
            self.data.insert_or_update_with(item, 1u, |_, v| *v += 1);
        }
    }
}

impl<K: Clone + Hash + TotalEq> FromIterator<K> for CountedSet<K> {
    fn from_iter<I: Iterator<K>>(iter: I) -> CountedSet<K> {
        let mut counts: CountedSet<K> = CountedSet::new();
        counts.extend(iter);
        counts
    }
}

impl<K: Hash + TotalEq + Show> Show for CountedSet<K> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.data.fmt(f)
    }
}

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
    let mut elems: Vec<_> = count.elements().iter().map(|&x|x).collect();
    elems.sort();
    assert!(elems.as_slice() == strings);
}

#[test]
fn test_most_common() {
    let strings = ["a", "b", "a", "b", "b", "c"];
    let answer = [("b", 3u), ("a", 2)];

    let count: CountedSet<_> = strings.iter().map(|&x|x).collect();
    let v: Vec<_> = count.most_common(2);
    assert!(answer == v.as_slice())
}

#[test]
fn test_get() {
    let strings = ["a", "b", "a", "b", "b", "c"];
    let count: CountedSet<_> = strings.iter().map(|&x|x).collect();
    assert_eq!(count.get(&"a"), &2)
    assert_eq!(count.get(&"b"), &3)
    assert_eq!(count.get(&"c"), &1)
    assert_eq!(count.get(&"missing"), &0)
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
