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

use iter::Iterator;
use option::{Option, Some, None};

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

/// A trait to represent constructable containers
pub trait NewContainer {
    /// Creates a new container.
    fn new() -> Self { NewContainer::with_capacity(0) }

    /// Creates a new container with a capacity of `capacity`.
    fn with_capacity(capacity: uint) -> Self;
}

/// A trait to represent mutable containers
pub trait Mutable: Container {
    /// Clear the container, removing all values.
    fn clear(&mut self);
}

/// A trait to represent an order sequence of values, where values are looked
/// up by their position.
pub trait Seq<T>: Container {
}

/// A trait that provides basic operations to modify the contents of a
/// sequence.
pub trait MutableSeq<T>: Seq<T> + Mutable {
    /// Append an element to a sequence
    fn push(&mut self, value: T);

    /// Moving all elements into the current sequence. This does not copy any
    /// elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut a = ~[~1];
    /// a.push_all((~[~2, ~3, ~4]).move_iter());
    /// assert!(a == ~[~1, ~2, ~3, ~4]);
    /// ```
    #[inline]
    fn push_all<Iter: Iterator<T>>(&mut self, mut iter: Iter) {
        for x in &mut iter {
            self.push(x);
        }
    }

    /// Remove the last element from a sequence and return it, failing if it is
    /// empty
    #[inline]
    fn pop(&mut self) -> T {
        self.pop_opt().expect("pop: empty sequence")
    }

    /// Remove the last element from a sequence and return it, or `None` if it
    /// is empty
    fn pop_opt(&mut self) -> Option<T>;

    /// Removes the first element from a sequence and return it
    #[inline]
    fn shift(&mut self) -> T {
        self.shift_opt().expect("shift: empty sequence")
    }

    /// Removes the first element from a sequence and return it, or `None` if
    /// it is empty
    fn shift_opt(&mut self) -> Option<T>;

    /// Prepend an element to the sequence
    fn unshift(&mut self, x: T);

    /// Insert an element at position i within v, shifting all
    /// elements after position i one position to the right.
    fn insert(&mut self, i: uint, x: T);

    /// Remove and return the element at position i within v, shifting
    /// all elements after position i one position to the left.
    fn remove(&mut self, i: uint) -> T;
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
    fn swap(&mut self, k: K, v: V) -> Option<V>;

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, k: &K) -> Option<V>;

    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V>;
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
