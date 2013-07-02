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


use std::cmp;
use std::iterator::{Iterator,IteratorUtil,ZipIterator,Counter,EnumerateIterator,FilterMapIterator};
use std::uint;
use std::util::replace;
use std::vec::{VecIterator,VecMutIterator,VecRevIterator,VecMutRevIterator};
use std::vec::VecConsumeIterator;

#[allow(missing_doc)]
pub struct SmallIntMap<T> {
    priv v: ~[Option<T>],
}

impl<V> Container for SmallIntMap<V> {
    /// Return the number of elements in the map
    fn len(&self) -> uint {
        let mut sz = 0;
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
                Some(_) => sz += 1,
                None => {}
            }
        }
        sz
    }

    /// Return true if the map contains no elements
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl<V> Mutable for SmallIntMap<V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.v.clear() }
}

impl<V> Map<uint, V> for SmallIntMap<V> {
    /// Return true if the map contains a value for the specified key
    fn contains_key(&self, key: &uint) -> bool {
        self.find(key).is_some()
    }

    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &uint) -> Option<&'a V> {
        if *key < self.v.len() {
            match self.v[*key] {
              Some(ref value) => Some(value),
              None => None
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
              Some(ref mut value) => Some(value),
              None => None
            }
        } else {
            None
        }
    }

    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    fn insert(&mut self, key: uint, value: V) -> bool {
        let exists = self.contains_key(&key);
        let len = self.v.len();
        if len <= key {
            self.v.grow_fn(key - len + 1, |_| None);
        }
        self.v[key] = Some(value);
        !exists
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    fn remove(&mut self, key: &uint) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, key: uint, value: V) -> Option<V> {
        match self.find_mut(&key) {
            Some(loc) => { return Some(replace(loc, value)); }
            None => ()
        }
        self.insert(key, value);
        return None;
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, key: &uint) -> Option<V> {
        if *key >= self.v.len() {
            return None;
        }
        replace(&mut self.v[*key], None)
    }
}

impl<V> SmallIntMap<V> {
    /// Create an empty SmallIntMap
    pub fn new() -> SmallIntMap<V> { SmallIntMap{v: ~[]} }

    /// Visit all key-value pairs in order
    pub fn each<'a>(&'a self, it: &fn(&uint, &'a V) -> bool) -> bool {
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
              Some(ref elt) => if !it(&i, elt) { return false; },
              None => ()
            }
        }
        return true;
    }

    /// Visit all keys in order
    pub fn each_key(&self, blk: &fn(key: &uint) -> bool) -> bool {
        self.each(|k, _| blk(k))
    }

    /// Visit all values in order
    pub fn each_value<'a>(&'a self, blk: &fn(value: &'a V) -> bool) -> bool {
        self.each(|_, v| blk(v))
    }

    /// Iterate over the map and mutate the contained values
    pub fn mutate_values(&mut self, it: &fn(&uint, &mut V) -> bool) -> bool {
        for uint::range(0, self.v.len()) |i| {
            match self.v[i] {
              Some(ref mut elt) => if !it(&i, elt) { return false; },
              None => ()
            }
        }
        return true;
    }

    /// Visit all key-value pairs in reverse order
    pub fn each_reverse<'a>(&'a self, it: &fn(uint, &'a V) -> bool) -> bool {
        for uint::range_rev(self.v.len(), 0) |i| {
            match self.v[i] {
              Some(ref elt) => if !it(i, elt) { return false; },
              None => ()
            }
        }
        return true;
    }

    pub fn get<'a>(&'a self, key: &uint) -> &'a V {
        self.find(key).expect("key not present")
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys.
    /// Iterator element type is (uint, &'r V)
    pub fn iter<'r>(&'r self) -> SmallIntMapIterator<'r, V> {
        SmallIntMapIterator {
            iter: Counter::new(0,1).zip(self.v.iter())
        }
    }

    /// An iterator visiting all key-value pairs in ascending order by the keys,
    /// with mutable references to the values
    /// Iterator element type is (uint, &'r mut V)
    pub fn mut_iter<'r>(&'r mut self) -> SmallIntMapMutIterator<'r, V> {
        SmallIntMapMutIterator {
            iter: Counter::new(0,1).zip(self.v.mut_iter())
        }
    }

    /// An iterator visiting all key-value pairs in descending order by the keys.
    /// Iterator element type is (uint, &'r V)
    pub fn rev_iter<'r>(&'r self) -> SmallIntMapRevIterator<'r, V> {
        SmallIntMapRevIterator {
            iter: Counter::new(self.len() as int - 1, -1).zip(self.v.rev_iter())
        }
    }

    /// An iterator visiting all key-value pairs in descending order by the keys,
    /// with mutable references to the values
    /// Iterator element type is (uint, &'r mut V)
    pub fn mut_rev_iter<'r>(&'r mut self) -> SmallIntMapMutRevIterator <'r, V> {
        SmallIntMapMutRevIterator {
            iter: Counter::new(self.len() as int - 1, -1).zip(self.v.mut_rev_iter())
        }
    }

    /// Empties the hash map, moving all values into the specified closure
    pub fn consume(&mut self)
        -> FilterMapIterator<(uint, Option<V>), (uint, V),
                EnumerateIterator<Option<V>, VecConsumeIterator<Option<V>>>>
    {
        let values = replace(&mut self.v, ~[]);
        values.consume_iter().enumerate().filter_map(|(i, v)| {
            v.map_consume(|v| (i, v))
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


macro_rules! iterator {
    /* FIXME: #4375 Cannot attach documentation/attributes to a macro generated struct.
    (struct $name:ident -> $ptr:ty, $elem:ty) => {
        pub struct $name<'self, T> {
            priv ptr: $ptr,
            priv end: $ptr,
            priv lifetime: $elem // FIXME: #5922
        }
    };*/
    (impl $name:ident -> $elem:ty) => {
        impl<'self, T> Iterator<(uint, $elem)> for $name<'self, T> {
            #[inline]
            pub fn next(&mut self) -> Option<(uint, $elem)> {
                for self.iter.advance |(idx, elem)| {
                    match elem {
                        &None => {}
                        &Some(ref e) => { return Some((idx as uint, e)) }
                    }
                }

                None
            }
        }
    }
}

macro_rules! mut_iterator {
    /* FIXME: #4375 Cannot attach documentation/attributes to a macro generated struct.
    (struct $name:ident -> $ptr:ty, $elem:ty) => {
        pub struct $name<'self, T> {
            priv ptr: $ptr,
            priv end: $ptr,
            priv lifetime: $elem // FIXME: #5922
        }
    };*/
    (impl $name:ident -> $elem:ty) => {
        impl<'self, T> Iterator<(uint, $elem)> for $name<'self, T> {
            #[inline]
            pub fn next(&mut self) -> Option<(uint, $elem)> {
                for self.iter.advance |(idx, elem)| {
                    match elem {
                        &None => {}
                        &Some(ref mut e) => { return Some((idx as uint, e)) }
                    }
                }

                None
            }
        }
    }
}

pub struct SmallIntMapIterator<'self, T> {
    priv iter: ZipIterator<int,
                           Counter<int>,
                           &'self Option<T>,
                           VecIterator<'self, Option<T> > >
}

iterator!{impl SmallIntMapIterator -> &'self T}

pub struct SmallIntMapMutIterator<'self, T> {
    priv iter: ZipIterator<int,
                           Counter<int>,
                           &'self mut Option<T>,
                           VecMutIterator<'self, Option<T> > >
}

mut_iterator!{impl SmallIntMapMutIterator -> &'self mut T}

pub struct SmallIntMapRevIterator<'self, T> {
    priv iter: ZipIterator<int,
                           Counter<int>,
                           &'self Option<T>,
                           VecRevIterator<'self, Option<T> > >
}

iterator!{impl SmallIntMapRevIterator -> &'self T}

pub struct SmallIntMapMutRevIterator<'self, T> {
    priv iter: ZipIterator<int,
                           Counter<int>,
                           &'self mut Option<T>,
                           VecMutRevIterator<'self, Option<T> > >
}

mut_iterator!{impl SmallIntMapMutRevIterator -> &'self mut T}

/// A set implemented on top of the SmallIntMap type. This set is always a set
/// of integers, and the space requirements are on the order of the highest
/// valued integer in the set.
pub struct SmallIntSet {
    priv map: SmallIntMap<()>
}

impl Container for SmallIntSet {
    /// Return the number of elements in the map
    fn len(&self) -> uint {
        self.map.len()
    }

    /// Return true if the map contains no elements
    fn is_empty(&self) -> bool { self.len() == 0 }
}

impl Mutable for SmallIntSet {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) { self.map.clear() }
}

impl Set<uint> for SmallIntSet {
    /// Return true if the set contains a value
    fn contains(&self, value: &uint) -> bool { self.map.contains_key(value) }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty uintersection.
    fn is_disjoint(&self, other: &SmallIntSet) -> bool {
        for self.each |v| { if other.contains(v) { return false } }
        true
    }

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &SmallIntSet) -> bool {
        for self.each |v| { if !other.contains(v) { return false } }
        true
    }

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &SmallIntSet) -> bool {
        other.is_subset(self)
    }

    /// Visit the values representing the difference
    fn difference(&self, other: &SmallIntSet, f: &fn(&uint) -> bool) -> bool {
        self.each(|v| other.contains(v) || f(v))
    }

    /// Visit the values representing the symmetric difference
    fn symmetric_difference(&self,
                            other: &SmallIntSet,
                            f: &fn(&uint) -> bool) -> bool {
        let len = cmp::max(self.map.v.len() ,other.map.v.len());

        for uint::range(0, len) |i| {
            if self.contains(&i) ^ other.contains(&i) {
                if !f(&i) { return false; }
            }
        }
        return true;
    }

    /// Visit the values representing the uintersection
    fn intersection(&self, other: &SmallIntSet, f: &fn(&uint) -> bool) -> bool {
        self.each(|v| !other.contains(v) || f(v))
    }

    /// Visit the values representing the union
    fn union(&self, other: &SmallIntSet, f: &fn(&uint) -> bool) -> bool {
        let len = cmp::max(self.map.v.len() ,other.map.v.len());

        for uint::range(0, len) |i| {
            if self.contains(&i) || other.contains(&i) {
                if !f(&i) { return false; }
            }
        }
        return true;
    }
}

impl MutableSet<uint> for SmallIntSet {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: uint) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &uint) -> bool { self.map.remove(value) }
}

impl SmallIntSet {
    /// Create an empty SmallIntSet
    pub fn new() -> SmallIntSet { SmallIntSet{map: SmallIntMap::new()} }

    /// Visit all values in order
    pub fn each(&self, f: &fn(&uint) -> bool) -> bool { self.map.each_key(f) }

    /// An iterator visiting all set members in ascending order.
    /// Iterator element type is uint
    pub fn iter<'r>(&'r self) -> SmallIntSetIterator<'r> {
        SmallIntSetIterator {
            iter: self.map.iter()
        }
    }

    /// An iterator visiting all set members in descending order.
    /// Iterator element type is uint
    pub fn rev_iter<'r>(&'r mut self) -> SmallIntSetRevIterator<'r> {
        SmallIntSetRevIterator {
            iter: self.map.rev_iter()
        }
    }

}

pub struct SmallIntSetIterator<'self> {
    priv iter: SmallIntMapIterator<'self, ()>
}

pub struct SmallIntSetRevIterator<'self> {
    priv iter: SmallIntMapRevIterator<'self,()>
}

impl<'self> Iterator<uint> for SmallIntSetIterator<'self> {
    #[inline]
    pub fn next(&mut self) -> Option<uint> {
        let next_opt = self.iter.next();
        match next_opt {
            None => { None }
            Some((idx, _)) => { Some(idx) }
        }
    }
}

impl<'self> Iterator<uint> for SmallIntSetRevIterator<'self> {
    #[inline]
    pub fn next(&mut self) -> Option<uint> {
        let next_opt = self.iter.next();
        match next_opt {
            None => { None }
            Some((idx, _)) => { Some(idx) }
        }
    }
}



#[cfg(test)]
mod tests {

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
        assert_eq!(map.find(&3).get(), &10);
        assert_eq!(map.find(&5).get(), &3);
        assert_eq!(map.find(&9).get(), &1);

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
    fn test_pop() {
        let mut m = SmallIntMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_iterator() {
        let mut a = SmallIntMap::new();

        assert!(a.insert(0,1));
        assert!(a.insert(1,2));
        assert!(a.insert(2,5));
        assert!(a.insert(3,10));
        assert!(a.insert(4,11));

        let mut it = a.iter();
        assert_eq!(it.next().unwrap(), (0, &1));
        assert_eq!(it.next().unwrap(), (1, &2));
        assert_eq!(it.next().unwrap(), (2, &5));
        assert_eq!(it.next().unwrap(), (3, &10));
        assert_eq!(it.next().unwrap(), (4, &11));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_mut_iterator() {
        let mut a = SmallIntMap::new();

        assert!(a.insert(0,1));
        assert!(a.insert(1,1));
        assert!(a.insert(2,1));
        assert!(a.insert(3,1));
        assert!(a.insert(4,1));

        for a.mut_iter().advance |(_,v)| {
            *v += 1;
        }

       assert!(a.iter().all(|(_,v)| *v == 2));
    }

    #[test]
    fn test_rev_iterator() {
       let mut a = SmallIntMap::new();

        assert!(a.insert(0,1));
        assert!(a.insert(1,2));
        assert!(a.insert(2,5));
        assert!(a.insert(3,10));
        assert!(a.insert(4,11));

        let mut b = SmallIntMap::new();

        assert!(b.insert(0,11));
        assert!(b.insert(1,10));
        assert!(b.insert(2,5));
        assert!(b.insert(3,2));
        assert!(b.insert(4,1));

        let (a_it, b_it) = (a.iter(), b.rev_iter());

        assert!(a_it.zip(b_it).all(|( (_ ,v1), (_, v2) )| *v1 == *v2));
    }

    #[test]
    fn test_mut_rev_iterator() {
        let mut a = SmallIntMap::new();

        assert!(a.insert(0,5));
        assert!(a.insert(1,4));
        assert!(a.insert(2,3));
        assert!(a.insert(3,2));
        assert!(a.insert(4,1));

        for a.mut_rev_iter().advance |(i,v)| {
            *v += i as int;
        }

        assert!(a.iter().all(|(_,v)| *v == 5 ));
    }

    #[test]
    fn test_consume() {
        let mut m = SmallIntMap::new();
        m.insert(1, ~2);
        let mut called = false;
        for m.consume().advance |(k, v)| {
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
mod test_set {

    use super::SmallIntSet;

    #[test]
    fn test_disjoint() {
        let mut xs = SmallIntSet::new();
        let mut ys = SmallIntSet::new();
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(5));
        assert!(ys.insert(11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(xs.insert(7));
        assert!(xs.insert(19));
        assert!(xs.insert(4));
        assert!(ys.insert(2));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = SmallIntSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = SmallIntSet::new();
        assert!(b.insert(0));
        assert!(b.insert(7));
        assert!(b.insert(19));
        assert!(b.insert(250));
        assert!(b.insert(11));
        assert!(b.insert(200));

        assert!(!a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(!b.is_superset(&a));

        assert!(b.insert(5));

        assert!(a.is_subset(&b));
        assert!(!a.is_superset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn test_intersection() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for a.intersection(&b) |x| {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for a.difference(&b) |x| {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [1, 5, 11, 14, 22];
        for a.symmetric_difference(&b) |x| {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = SmallIntSet::new();
        let mut b = SmallIntSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [1, 3, 5, 9, 11, 13, 16, 19, 24];
        for a.union(&b) |x| {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_iterator() {
        let mut a = SmallIntSet::new();

        assert!(a.insert(0));
        assert!(a.insert(1));
        assert!(a.insert(2));
        assert!(a.insert(3));
        assert!(a.insert(4));

        let mut it = a.iter();
        assert_eq!(it.next().unwrap(), 0);
        assert_eq!(it.next().unwrap(), 1);
        assert_eq!(it.next().unwrap(), 2);
        assert_eq!(it.next().unwrap(), 3);
        assert_eq!(it.next().unwrap(), 4);
        assert!(it.next().is_none());
    }

    #[test]
    fn test_rev_iterator() {
       let mut a = SmallIntSet::new();

        assert!(a.insert(0));
        assert!(a.insert(1));
        assert!(a.insert(2));
        assert!(a.insert(3));
        assert!(a.insert(4));

        let a_it = a.rev_iter();

        assert!(do a_it.enumerate().all |( i, v2 )| {
            i + v2 == 4
        });
    }

}
