// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * A simple map based on a vector for small integer keys. Space requirements
 * are O(highest integer key).
 */

#[allow(missing_doc)];

use std::iter::{FilterMap, Invert};
use std::util::replace;
use std::vec::{VecIterator, VecMutIterator};
use std::vec;

#[allow(missing_doc)]
pub struct SmallIntMap<T> {
    priv v: ~[(uint, Option<T>)],
}

impl<V> Container for SmallIntMap<V> {
    /// Return the number of elements in the map
    fn len(&self) -> uint {
        self.v.iter().count(|&(_, ref elt)| elt.is_some())
    }

    /// Return true if there are no elements in the map
    fn is_empty(&self) -> bool {
        self.v.iter().all(|&(_, ref elt)| elt.is_none())
    }
}

impl<V> Mutable for SmallIntMap<V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> Map<uint, V> for SmallIntMap<V> {
    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &uint) -> Option<&'a V> {
        if *key < self.v.len() {
            match self.v[*key] {
              (_, Some(ref value)) => Some(value),
              (_, None) => None
            }
        } else {
            None
        }
    }
}

impl<V> MutableMap<uint, V> for SmallIntMap<V> {
    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &uint) -> Option<&'a mut V> {
        if *key < self.v.len() {
            match self.v[*key] {
              (_, Some(ref mut value)) => Some(value),
              (_, None) => None
            }
        } else {
            None
        }
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    fn find_or_insert_with<'a>(&'a mut self, key: uint, f: &fn(&uint) -> V)
                               -> (&'a uint, &'a mut V) {
        let len = self.v.len();
        if len <= key {
            self.v.grow_fn(key - len + 1, |i| (i + len, None));
        }
        match self.v[key] {
            (ref key, Some(ref mut value)) => (key, value),
            (ref key, ref mut e) => {
                let value = f(key);
                *e = Some(value);
                (key, e.get_mut_ref())
            }
        }
    }

    /// Does nothing for this implementation.
    fn reserve_at_least(&mut self, _: uint) {
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    fn remove(&mut self, key: &uint) -> bool {
        self.pop(key).is_some()
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, key: &uint) -> Option<V> {
        if *key >= self.v.len() {
            return None;
        }
        let (_, ref mut e) = self.v[*key];
        e.take()
    }
}

impl<V> SmallIntMap<V> {
    /// Create an empty SmallIntMap
    pub fn new() -> SmallIntMap<V> { SmallIntMap{v: ~[]} }

    pub fn get<'a>(&'a self, key: &uint) -> &'a V {
        self.find(key).expect("key not present")
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys.
    /// Iterator element type is (uint, &'r V)
    pub fn iter<'r>(&'r self) -> SmallIntMapIterator<'r, V> {
        SmallIntMapIterator {
            front: 0,
            back: self.v.len(),
            iter: self.v.iter()
        }
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys,
    /// with mutable references to the values
    /// Iterator element type is (uint, &'r mut V)
    pub fn mut_iter<'r>(&'r mut self) -> SmallIntMapMutIterator<'r, V> {
        SmallIntMapMutIterator {
            front: 0,
            back: self.v.len(),
            iter: self.v.mut_iter()
        }
    }

    /// An iterator visiting all key-value pairs in descending order by the keys.
    /// Iterator element type is (uint, &'r V)
    pub fn rev_iter<'r>(&'r self) -> SmallIntMapRevIterator<'r, V> {
        self.iter().invert()
    }

    /// An iterator visiting all key-value pairs in descending order by the keys,
    /// with mutable references to the values
    /// Iterator element type is (uint, &'r mut V)
    pub fn mut_rev_iter<'r>(&'r mut self) -> SmallIntMapMutRevIterator <'r, V> {
        self.mut_iter().invert()
    }

    /// Empties the hash map, moving all values into the specified closure
    pub fn move_iter(&mut self)
        -> FilterMap<(uint, Option<V>), (uint, V),
                vec::MoveIterator<(uint, Option<V>)>>
    {
        let values = replace(&mut self.v, ~[]);
        values.move_iter().filter_map(|(i, v)| {
            v.map(|v| (i, v))
        })
    }
}

impl<V:Clone> SmallIntMap<V> {
    pub fn update_with_key(&mut self, key: uint, val: V,
                           ff: &fn(uint, V, V) -> V) -> bool {
        let new_val = match self.find(&key) {
            None => val,
            Some(orig) => ff(key, (*orig).clone(), val)
        };
        self.insert(key, new_val)
    }

    pub fn update(&mut self, key: uint, newval: V, ff: &fn(V, V) -> V)
                  -> bool {
        self.update_with_key(key, newval, |_k, v, v1| ff(v,v1))
    }
}

pub struct SmallIntMapIterator<'self, T> {
    priv front: uint,
    priv back: uint,
    priv iter: VecIterator<'self, (uint, Option<T>)>
}

impl<'self, T> Iterator<(uint, &'self T)> for SmallIntMapIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<(uint, &'self T)> {
        while self.front < self.back {
            match self.iter.next() {
                Some(elem) => {
                    let (_, ref v) = *elem;
                    if v.is_some() {
                        let index = self.front;
                        self.front += 1;
                        return Some((index, v.get_ref()));
                    }
                }
                _ => ()
            }
            self.front += 1;
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.back - self.front))
    }
}

impl<'self, T> DoubleEndedIterator<(uint, &'self T)> for SmallIntMapIterator<'self, T> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, &'self T)> {
        while self.front < self.back {
            match self.iter.next_back() {
                Some(elem) => {
                    let (_, ref v) = *elem;
                    if v.is_some() {
                        self.back -= 1;
                        return Some((self.back, v.get_ref()));
                    }
                }
                _ => ()
            }
            self.back -= 1;
        }
        None
    }
}

pub type SmallIntMapRevIterator<'self, T> = Invert<SmallIntMapIterator<'self, T>>;

pub struct SmallIntMapMutIterator<'self, T> {
    priv front: uint,
    priv back: uint,
    priv iter: VecMutIterator<'self, (uint, Option<T>)>
}

impl<'self, T> Iterator<(uint, &'self mut T)> for SmallIntMapMutIterator<'self, T> {
    #[inline]
    fn next(&mut self) -> Option<(uint, &'self mut T)> {
        while self.front < self.back {
            match self.iter.next() {
                Some(elem) => {
                    let (_, ref mut v) = *elem;
                    if v.is_some() {
                        let index = self.front;
                        self.front += 1;
                        return Some((index, v.get_mut_ref()));
                    }
                }
                _ => ()
            }
            self.front += 1;
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        (0, Some(self.back - self.front))
    }
}

impl<'self, T> DoubleEndedIterator<(uint, &'self mut T)> for SmallIntMapMutIterator<'self, T> {
    #[inline]
    fn next_back(&mut self) -> Option<(uint, &'self mut T)> {
        while self.front < self.back {
            match self.iter.next_back() {
                Some(elem) => {
                    let (_, ref mut v) = *elem;
                    if v.is_some() {
                        self.back -= 1;
                        return Some((self.back, v.get_mut_ref()));
                    }
                }
                _ => ()
            }
            self.back -= 1;
        }
        None
    }
}

pub type SmallIntMapMutRevIterator<'self, T> = Invert<SmallIntMapMutIterator<'self, T>>;

#[cfg(test)]
mod test_map {

    use super::SmallIntMap;

    #[test]
    fn test_find_mut() {
        let mut m = SmallIntMap::new();
        assert!(m.insert(1, 12));
        assert!(m.insert(2, 8));
        assert!(m.insert(5, 14));
        let new = 100;
        match m.find_mut(&5) {
            None => fail!(), Some(x) => *x = new
        }
        assert_eq!(m.find(&5), Some(&new));
    }

    #[test]
    fn test_len() {
        let mut map = SmallIntMap::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.insert(5, 20));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert!(map.insert(11, 12));
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());
        assert!(map.insert(14, 22));
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut map = SmallIntMap::new();
        assert!(map.insert(5, 20));
        assert!(map.insert(11, 12));
        assert!(map.insert(14, 22));
        map.clear();
        assert!(map.is_empty());
        assert!(map.find(&5).is_none());
        assert!(map.find(&11).is_none());
        assert!(map.find(&14).is_none());
    }

    #[test]
    fn test_insert_with_key() {
        let mut map = SmallIntMap::new();

        // given a new key, initialize it with this new count, given
        // given an existing key, add more to its count
        fn addMoreToCount(_k: uint, v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        fn addMoreToCount_simple(v0: uint, v1: uint) -> uint {
            v0 + v1
        }

        // count integers
        map.update(3, 1, addMoreToCount_simple);
        map.update_with_key(9, 1, addMoreToCount);
        map.update(3, 7, addMoreToCount_simple);
        map.update_with_key(5, 3, addMoreToCount);
        map.update_with_key(3, 2, addMoreToCount);

        // check the total counts
        assert_eq!(map.find(&3).unwrap(), &10);
        assert_eq!(map.find(&5).unwrap(), &3);
        assert_eq!(map.find(&9).unwrap(), &1);

        // sadly, no sevens were counted
        assert!(map.find(&7).is_none());
    }

    #[test]
    fn test_swap() {
        let mut m = SmallIntMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }


    #[test]
    fn test_find_or_insert() {
        let mut m = SmallIntMap::new();
        {
            let (k, v) = m.find_or_insert(1, 2);
            assert_eq!((*k, *v), (1, 2));
        }
        {
            let (k, v) = m.find_or_insert(1, 3);
            assert_eq!((*k, *v), (1, 2));
        }
    }

    #[test]
    fn test_find_or_insert_with() {
        let mut m = SmallIntMap::new();
        {
            let (k, v) = m.find_or_insert_with(1, |_| 2);
            assert_eq!((*k, *v), (1, 2));
        }
        {
            let (k, v) = m.find_or_insert_with(1, |_| 3);
            assert_eq!((*k, *v), (1, 2));
        }
    }

    #[test]
    fn test_insert_or_update_with() {
        let mut m = SmallIntMap::new();
        assert_eq!(*m.insert_or_update_with(1, 2, |_,x| *x+=1), 2);
        assert_eq!(*m.insert_or_update_with(1, 2, |_,x| *x+=1), 3);
    }


    #[test]
    fn test_pop() {
        let mut m = SmallIntMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        let mut it = m.iter();
        assert_eq!(it.size_hint(), (0, Some(11)));
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.size_hint(), (0, Some(10)));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.size_hint(), (0, Some(9)));
        assert_eq!(it.next().unwrap(), (3, &5));
        assert_eq!(it.size_hint(), (0, Some(7)));
        assert_eq!(it.next().unwrap(), (6, &10));
        assert_eq!(it.size_hint(), (0, Some(4)));
        assert_eq!(it.next().unwrap(), (10, &11));
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_iterator_size_hints() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        assert_eq!(m.iter().size_hint(), (0, Some(11)));
        assert_eq!(m.rev_iter().size_hint(), (0, Some(11)));
        assert_eq!(m.mut_iter().size_hint(), (0, Some(11)));
        assert_eq!(m.mut_rev_iter().size_hint(), (0, Some(11)));
    }

    #[test]
    fn test_mut_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        for (k, v) in m.mut_iter() {
            *v += k as int;
        }

        let mut it = m.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &3));
        assert_eq!(it.next().unwrap(), (3, &8));
        assert_eq!(it.next().unwrap(), (6, &16));
        assert_eq!(it.next().unwrap(), (10, &21));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_rev_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        let mut it = m.rev_iter();
        assert_eq!(it.next().unwrap(), (10, &11));
        assert_eq!(it.next().unwrap(), (6, &10));
        assert_eq!(it.next().unwrap(), (3, &5));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.next().unwrap(), (0, &1));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_mut_rev_iterator() {
        let mut m = SmallIntMap::new();

        assert!(m.insert(0, 1));
        assert!(m.insert(1, 2));
        assert!(m.insert(3, 5));
        assert!(m.insert(6, 10));
        assert!(m.insert(10, 11));

        for (k, v) in m.mut_rev_iter() {
            *v += k as int;
        }

        let mut it = m.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &3));
        assert_eq!(it.next().unwrap(), (3, &8));
        assert_eq!(it.next().unwrap(), (6, &16));
        assert_eq!(it.next().unwrap(), (10, &21));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_move_iter() {
        let mut m = SmallIntMap::new();
        m.insert(1, ~2);
        let mut called = false;
        for (k, v) in m.move_iter() {
            assert!(!called);
            called = true;
            assert_eq!(k, 1);
            assert_eq!(v, ~2);
        }
        assert!(called);
        m.insert(2, ~1);
    }
}

#[cfg(test)]
mod bench {

    use super::*;
    use test::BenchHarness;
    use container::bench::*;

    // Find seq
    #[bench]
    pub fn insert_rand_100(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_rand_n(100, &mut m, bh);
    }

    #[bench]
    pub fn insert_rand_10_000(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_rand_n(10_000, &mut m, bh);
    }

    // Insert seq
    #[bench]
    pub fn insert_seq_100(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_seq_n(100, &mut m, bh);
    }

    #[bench]
    pub fn insert_seq_10_000(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        insert_seq_n(10_000, &mut m, bh);
    }

    // Find rand
    #[bench]
    pub fn find_rand_100(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_rand_n(100, &mut m, bh);
    }

    #[bench]
    pub fn find_rand_10_000(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_rand_n(10_000, &mut m, bh);
    }

    // Find seq
    #[bench]
    pub fn find_seq_100(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_seq_n(100, &mut m, bh);
    }

    #[bench]
    pub fn find_seq_10_000(bh: &mut BenchHarness) {
        let mut m : SmallIntMap<uint> = SmallIntMap::new();
        find_seq_n(10_000, &mut m, bh);
    }
}
