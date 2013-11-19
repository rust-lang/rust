// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Container traits

use cell::Cell;
use option::Option;
use util;

/// A trait to represent the abstract idea of a container. The only concrete
/// knowledge known is the number of elements contained within.
pub trait Container {
    /// Return the number of elements in the container
    fn len(&self) -> uint;

    /// Return true if the container contains no elements
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A trait to represent mutable containers
pub trait Mutable: Container {
    /// Clear the container, removing all values.
    fn clear(&mut self);
}

/// A map is a key-value store where values may be looked up by their keys. This
/// trait provides basic operations to operate on these stores.
pub trait Map<K, V>: Container {
    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, key: &K) -> Option<&'a V>;

    /// Return true if the map contains a value for the specified key
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_some()
    }
}

/// This trait provides basic operations to modify the contents of a map.
pub trait MutableMap<K, V>: Map<K, V> + Mutable {
    /// Insert a key-value pair into the map. An existing value for a
    /// key is replaced by the new value. Return true if the key did
    /// not already exist in the map.
    #[inline]
    fn insert(&mut self, key: K, value: V) -> bool {
        self.swap(key, value).is_none()
    }

    /// Remove a key-value pair from the map. Return true if the key
    /// was present in the map, otherwise false.
    #[inline]
    fn remove(&mut self, key: &K) -> bool {
        self.pop(key).is_some()
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    #[inline]
    fn swap(&mut self, k: K, v: V) -> Option<V> {
        let cell = Cell::new(v);
        let (_, r) = self.find_or_insert_with(k, |_| cell.take());
        cell.take_opt().map(|v| util::replace(r, v))
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, k: &K) -> Option<V>;

    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>;

    /// Return the value corresponding to the key in the map, or insert
    /// and return the value if it doesn't exist.
    #[inline]
    fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> (&'a K, &'a mut V) {
        let cell = Cell::new(v);
        self.find_or_insert_with(k, |_| cell.take())
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    fn find_or_insert_with<'a>(&'a mut self, k: K, f: &fn(&K) -> V)
                               -> (&'a K, &'a mut V);

    /// Insert a key-value pair into the map if the key is not already present.
    /// Otherwise, modify the existing value for the key.
    /// Returns the new or modified value for the key.
    #[inline]
    fn insert_or_update_with<'a>(&'a mut self, k: K, v: V,
                                     f: &fn(&K, &mut V)) -> &'a mut V {
        let cell = Cell::new(v);
        let (k, v) = self.find_or_insert_with(k, |_| cell.take());
        if !cell.is_empty() {
            f(k, v);
        }
        v
    }

    /// Reserve space for at least `n` elements in the data structure
    /// if applicable.  Does nothing if not pre-allocating space doesn't
    /// make sense for the underlying data structure.
    fn reserve_at_least(&mut self, n: uint);
}

/// A set is a group of objects which are each distinct from one another. This
/// trait represents actions which can be performed on sets to iterate over
/// them.
pub trait Set<T>: Container {
    /// Return true if the set contains a value
    fn contains(&self, value: &T) -> bool;

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    fn is_disjoint(&self, other: &Self) -> bool;

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &Self) -> bool;

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &Self) -> bool;

    // FIXME #8154: Add difference, sym. difference, intersection and union iterators
}

/// This trait represents actions which can be performed on sets to mutate
/// them.
pub trait MutableSet<T>: Set<T> + Mutable {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: T) -> bool;

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &T) -> bool;
}
