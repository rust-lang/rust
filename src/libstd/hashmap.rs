// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An unordered map and set type implemented as hash tables
//!
//! The tables use a keyed hash with new random keys generated for each container, so the ordering
//! of a set of keys in a hash table is randomized.

#[mutable_doc];

use container::{Container, Mutable, Map, MutableMap, Set, MutableSet};
use clone::Clone;
use cmp::{Eq, Equiv};
use hash::Hash;
use iter::{Iterator, FromIterator, Extendable};
use iter::{FilterMap, Chain, Repeat, Zip};
use num;
use option::{None, Option, Some};
use rand::RngUtil;
use rand;
use uint;
use util::{replace, unreachable};
use vec::{ImmutableVector, MutableVector, OwnedVector};
use vec;

static INITIAL_CAPACITY: uint = 32u; // 2^5

struct Bucket<K,V> {
    hash: uint,
    key: K,
    value: V,
}

/// A hash map implementation which uses linear probing along with the SipHash
/// hash function for internal state. This means that the order of all hash maps
/// is randomized by keying each hash map randomly on creation.
///
/// It is required that the keys implement the `Eq` and `Hash` traits, although
/// this can frequently be achieved by just implementing the `Eq` and
/// `IterBytes` traits as `Hash` is automatically implemented for types that
/// implement `IterBytes`.
pub struct HashMap<K,V> {
    priv k0: u64,
    priv k1: u64,
    priv resize_at: uint,
    priv size: uint,
    priv buckets: ~[Option<Bucket<K, V>>],
}

// We could rewrite FoundEntry to have type Option<&Bucket<K, V>>
// which would be nifty
enum SearchResult {
    FoundEntry(uint), FoundHole(uint), TableFull
}

#[inline]
fn resize_at(capacity: uint) -> uint {
    (capacity * 3) / 4
}

impl<K:Hash + Eq,V> HashMap<K, V> {
    #[inline]
    fn to_bucket(&self, h: uint) -> uint {
        // A good hash function with entropy spread over all of the
        // bits is assumed. SipHash is more than good enough.
        h % self.buckets.len()
    }

    #[inline]
    fn next_bucket(&self, idx: uint, len_buckets: uint) -> uint {
        (idx + 1) % len_buckets
    }

    #[inline]
    fn bucket_sequence(&self, hash: uint,
                       op: &fn(uint) -> bool) -> bool {
        let start_idx = self.to_bucket(hash);
        let len_buckets = self.buckets.len();
        let mut idx = start_idx;
        loop {
            if !op(idx) { return false; }
            idx = self.next_bucket(idx, len_buckets);
            if idx == start_idx {
                return true;
            }
        }
    }

    #[inline]
    fn bucket_for_key(&self, k: &K) -> SearchResult {
        let hash = k.hash_keyed(self.k0, self.k1) as uint;
        self.bucket_for_key_with_hash(hash, k)
    }

    #[inline]
    fn bucket_for_key_equiv<Q:Hash + Equiv<K>>(&self, k: &Q)
                                               -> SearchResult {
        let hash = k.hash_keyed(self.k0, self.k1) as uint;
        self.bucket_for_key_with_hash_equiv(hash, k)
    }

    #[inline]
    fn bucket_for_key_with_hash(&self,
                                hash: uint,
                                k: &K)
                             -> SearchResult {
        let mut ret = TableFull;
        do self.bucket_sequence(hash) |i| {
            match self.buckets[i] {
                Some(ref bkt) if bkt.hash == hash && *k == bkt.key => {
                    ret = FoundEntry(i); false
                },
                None => { ret = FoundHole(i); false }
                _ => true,
            }
        };
        ret
    }

    #[inline]
    fn bucket_for_key_with_hash_equiv<Q:Equiv<K>>(&self,
                                                  hash: uint,
                                                  k: &Q)
                                               -> SearchResult {
        let mut ret = TableFull;
        do self.bucket_sequence(hash) |i| {
            match self.buckets[i] {
                Some(ref bkt) if bkt.hash == hash && k.equiv(&bkt.key) => {
                    ret = FoundEntry(i); false
                },
                None => { ret = FoundHole(i); false }
                _ => true,
            }
        };
        ret
    }

    /// Expand the capacity of the array to the next power of two
    /// and re-insert each of the existing buckets.
    #[inline]
    fn expand(&mut self) {
        let new_capacity = self.buckets.len() * 2;
        self.resize(new_capacity);
    }

    /// Expands the capacity of the array and re-insert each of the
    /// existing buckets.
    fn resize(&mut self, new_capacity: uint) {
        self.resize_at = resize_at(new_capacity);

        let old_buckets = replace(&mut self.buckets,
                                  vec::from_fn(new_capacity, |_| None));

        self.size = 0;
        // move_rev_iter is more efficient
        for bucket in old_buckets.move_rev_iter() {
            self.insert_opt_bucket(bucket);
        }
    }

    fn insert_opt_bucket(&mut self, bucket: Option<Bucket<K, V>>) {
        match bucket {
            Some(Bucket{hash: hash, key: key, value: value}) => {
                self.insert_internal(hash, key, value);
            }
            None => {}
        }
    }

    #[inline]
    fn value_for_bucket<'a>(&'a self, idx: uint) -> &'a V {
        match self.buckets[idx] {
            Some(ref bkt) => &bkt.value,
            None => fail!("HashMap::find: internal logic error"),
        }
    }

    #[inline]
    fn mut_value_for_bucket<'a>(&'a mut self, idx: uint) -> &'a mut V {
        match self.buckets[idx] {
            Some(ref mut bkt) => &mut bkt.value,
            None => unreachable()
        }
    }

    /// Inserts the key value pair into the buckets.
    /// Assumes that there will be a bucket.
    /// True if there was no previous entry with that key
    fn insert_internal(&mut self, hash: uint, k: K, v: V) -> Option<V> {
        match self.bucket_for_key_with_hash(hash, &k) {
            TableFull => { fail!("Internal logic error"); }
            FoundHole(idx) => {
                self.buckets[idx] = Some(Bucket{hash: hash, key: k,
                                                value: v});
                self.size += 1;
                None
            }
            FoundEntry(idx) => {
                match self.buckets[idx] {
                    None => { fail!("insert_internal: Internal logic error") }
                    Some(ref mut b) => {
                        b.hash = hash;
                        b.key = k;
                        Some(replace(&mut b.value, v))
                    }
                }
            }
        }
    }

    fn pop_internal(&mut self, hash: uint, k: &K) -> Option<V> {
        // Removing from an open-addressed hashtable
        // is, well, painful.  The problem is that
        // the entry may lie on the probe path for other
        // entries, so removing it would make you think that
        // those probe paths are empty.
        //
        // To address this we basically have to keep walking,
        // re-inserting entries we find until we reach an empty
        // bucket.  We know we will eventually reach one because
        // we insert one ourselves at the beginning (the removed
        // entry).
        //
        // I found this explanation elucidating:
        // http://www.maths.lse.ac.uk/Courses/MA407/del-hash.pdf
        let mut idx = match self.bucket_for_key_with_hash(hash, k) {
            TableFull | FoundHole(_) => return None,
            FoundEntry(idx) => idx
        };

        let len_buckets = self.buckets.len();
        let bucket = self.buckets[idx].take();

        let value = do bucket.map_move |bucket| {
            bucket.value
        };

        /* re-inserting buckets may cause changes in size, so remember
        what our new size is ahead of time before we start insertions */
        let size = self.size - 1;
        idx = self.next_bucket(idx, len_buckets);
        while self.buckets[idx].is_some() {
            let bucket = self.buckets[idx].take();
            self.insert_opt_bucket(bucket);
            idx = self.next_bucket(idx, len_buckets);
        }
        self.size = size;

        value
    }
}

impl<K:Hash + Eq,V> Container for HashMap<K, V> {
    /// Return the number of elements in the map
    fn len(&self) -> uint { self.size }
}

impl<K:Hash + Eq,V> Mutable for HashMap<K, V> {
    /// Clear the map, removing all key-value pairs.
    fn clear(&mut self) {
        for bkt in self.buckets.mut_iter() {
            *bkt = None;
        }
        self.size = 0;
    }
}

impl<K:Hash + Eq,V> Map<K, V> for HashMap<K, V> {
    /// Return a reference to the value corresponding to the key
    fn find<'a>(&'a self, k: &K) -> Option<&'a V> {
        match self.bucket_for_key(k) {
            FoundEntry(idx) => Some(self.value_for_bucket(idx)),
            TableFull | FoundHole(_) => None,
        }
    }
}

impl<K:Hash + Eq,V> MutableMap<K, V> for HashMap<K, V> {
    /// Return a mutable reference to the value corresponding to the key
    fn find_mut<'a>(&'a mut self, k: &K) -> Option<&'a mut V> {
        let idx = match self.bucket_for_key(k) {
            FoundEntry(idx) => idx,
            TableFull | FoundHole(_) => return None
        };
        Some(self.mut_value_for_bucket(idx))
    }

    /// Insert a key-value pair from the map. If the key already had a value
    /// present in the map, that value is returned. Otherwise None is returned.
    fn swap(&mut self, k: K, v: V) -> Option<V> {
        // this could be faster.

        if self.size >= self.resize_at {
            // n.b.: We could also do this after searching, so
            // that we do not resize if this call to insert is
            // simply going to update a key in place.  My sense
            // though is that it's worse to have to search through
            // buckets to find the right spot twice than to just
            // resize in this corner case.
            self.expand();
        }

        let hash = k.hash_keyed(self.k0, self.k1) as uint;
        self.insert_internal(hash, k, v)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    fn pop(&mut self, k: &K) -> Option<V> {
        let hash = k.hash_keyed(self.k0, self.k1) as uint;
        self.pop_internal(hash, k)
    }
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    /// Create an empty HashMap
    pub fn new() -> HashMap<K, V> {
        HashMap::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty HashMap with space for at least `capacity`
    /// elements in the hash table.
    pub fn with_capacity(capacity: uint) -> HashMap<K, V> {
        let mut r = rand::task_rng();
        HashMap::with_capacity_and_keys(r.gen(), r.gen(), capacity)
    }

    /// Create an empty HashMap with space for at least `capacity`
    /// elements, using `k0` and `k1` as the keys.
    ///
    /// Warning: `k0` and `k1` are normally randomly generated, and
    /// are designed to allow HashMaps to be resistant to attacks that
    /// cause many collisions and very poor performance. Setting them
    /// manually using this function can expose a DoS attack vector.
    pub fn with_capacity_and_keys(k0: u64, k1: u64, capacity: uint) -> HashMap<K, V> {
        let cap = num::max(INITIAL_CAPACITY, capacity);
        HashMap {
            k0: k0, k1: k1,
            resize_at: resize_at(cap),
            size: 0,
            buckets: vec::from_fn(cap, |_| None)
        }
    }

    /// Reserve space for at least `n` elements in the hash table.
    pub fn reserve_at_least(&mut self, n: uint) {
        if n > self.buckets.len() {
            let buckets = n * 4 / 3 + 1;
            self.resize(uint::next_power_of_two(buckets));
        }
    }

    /// Modify and return the value corresponding to the key in the map, or
    /// insert and return a new value if it doesn't exist.
    pub fn mangle<'a,A>(&'a mut self, k: K, a: A, not_found: &fn(&K, A) -> V,
                        found: &fn(&K, &mut V, A)) -> &'a mut V {
        if self.size >= self.resize_at {
            // n.b.: We could also do this after searching, so
            // that we do not resize if this call to insert is
            // simply going to update a key in place.  My sense
            // though is that it's worse to have to search through
            // buckets to find the right spot twice than to just
            // resize in this corner case.
            self.expand();
        }

        let hash = k.hash_keyed(self.k0, self.k1) as uint;
        let idx = match self.bucket_for_key_with_hash(hash, &k) {
            TableFull => fail!("Internal logic error"),
            FoundEntry(idx) => { found(&k, self.mut_value_for_bucket(idx), a); idx }
            FoundHole(idx) => {
                let v = not_found(&k, a);
                self.buckets[idx] = Some(Bucket{hash: hash, key: k, value: v});
                self.size += 1;
                idx
            }
        };

        self.mut_value_for_bucket(idx)
    }

    /// Return the value corresponding to the key in the map, or insert
    /// and return the value if it doesn't exist.
    pub fn find_or_insert<'a>(&'a mut self, k: K, v: V) -> &'a mut V {
        self.mangle(k, v, |_k, a| a, |_k,_v,_a| ())
    }

    /// Return the value corresponding to the key in the map, or create,
    /// insert, and return a new value if it doesn't exist.
    pub fn find_or_insert_with<'a>(&'a mut self, k: K, f: &fn(&K) -> V)
                               -> &'a mut V {
        self.mangle(k, (), |k,_a| f(k), |_k,_v,_a| ())
    }

    /// Insert a key-value pair into the map if the key is not already present.
    /// Otherwise, modify the existing value for the key.
    /// Returns the new or modified value for the key.
    pub fn insert_or_update_with<'a>(&'a mut self, k: K, v: V,
                                     f: &fn(&K, &mut V)) -> &'a mut V {
        self.mangle(k, v, |_k,a| a, |k,v,_a| f(k,v))
    }

    /// Retrieves a value for the given key, failing if the key is not
    /// present.
    pub fn get<'a>(&'a self, k: &K) -> &'a V {
        match self.find(k) {
            Some(v) => v,
            None => fail!("No entry found for key: %?", k),
        }
    }

    /// Retrieves a (mutable) value for the given key, failing if the key
    /// is not present.
    pub fn get_mut<'a>(&'a mut self, k: &K) -> &'a mut V {
        match self.find_mut(k) {
            Some(v) => v,
            None => fail!("No entry found for key: %?", k),
        }
    }

    /// Return true if the map contains a value for the specified key,
    /// using equivalence
    pub fn contains_key_equiv<Q:Hash + Equiv<K>>(&self, key: &Q) -> bool {
        match self.bucket_for_key_equiv(key) {
            FoundEntry(_) => {true}
            TableFull | FoundHole(_) => {false}
        }
    }

    /// Return the value corresponding to the key in the map, using
    /// equivalence
    pub fn find_equiv<'a, Q:Hash + Equiv<K>>(&'a self, k: &Q)
                                             -> Option<&'a V> {
        match self.bucket_for_key_equiv(k) {
            FoundEntry(idx) => Some(self.value_for_bucket(idx)),
            TableFull | FoundHole(_) => None,
        }
    }

    /// Visit all keys
    pub fn each_key(&self, blk: &fn(k: &K) -> bool) -> bool {
        self.iter().advance(|(k, _)| blk(k))
    }

    /// Visit all values
    pub fn each_value<'a>(&'a self, blk: &fn(v: &'a V) -> bool) -> bool {
        self.iter().advance(|(_, v)| blk(v))
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// Iterator element type is (&'a K, &'a V).
    pub fn iter<'a>(&'a self) -> HashMapIterator<'a, K, V> {
        HashMapIterator { iter: self.buckets.iter() }
    }

    /// An iterator visiting all key-value pairs in arbitrary order,
    /// with mutable references to the values.
    /// Iterator element type is (&'a K, &'a mut V).
    pub fn mut_iter<'a>(&'a mut self) -> HashMapMutIterator<'a, K, V> {
        HashMapMutIterator { iter: self.buckets.mut_iter() }
    }

    /// Creates a consuming iterator, that is, one that moves each key-value
    /// pair out of the map in arbitrary order. The map cannot be used after
    /// calling this.
    pub fn move_iter(self) -> HashMapMoveIterator<K, V> {
        // `move_rev_iter` is more efficient than `move_iter` for vectors
        HashMapMoveIterator {iter: self.buckets.move_rev_iter()}
    }
}

impl<K: Hash + Eq, V: Clone> HashMap<K, V> {
    /// Like `find`, but returns a copy of the value.
    pub fn find_copy(&self, k: &K) -> Option<V> {
        self.find(k).map_move(|v| (*v).clone())
    }

    /// Like `get`, but returns a copy of the value.
    pub fn get_copy(&self, k: &K) -> V {
        (*self.get(k)).clone()
    }
}

impl<K:Hash + Eq,V:Eq> Eq for HashMap<K, V> {
    fn eq(&self, other: &HashMap<K, V>) -> bool {
        if self.len() != other.len() { return false; }

        do self.iter().all |(key, value)| {
            match other.find(key) {
                None => false,
                Some(v) => value == v
            }
        }
    }

    fn ne(&self, other: &HashMap<K, V>) -> bool { !self.eq(other) }
}

impl<K:Hash + Eq + Clone,V:Clone> Clone for HashMap<K,V> {
    fn clone(&self) -> HashMap<K,V> {
        let mut new_map = HashMap::with_capacity(self.len());
        for (key, value) in self.iter() {
            new_map.insert((*key).clone(), (*value).clone());
        }
        new_map
    }
}

/// HashMap iterator
#[deriving(Clone)]
pub struct HashMapIterator<'self, K, V> {
    priv iter: vec::VecIterator<'self, Option<Bucket<K, V>>>,
}

/// HashMap mutable values iterator
pub struct HashMapMutIterator<'self, K, V> {
    priv iter: vec::VecMutIterator<'self, Option<Bucket<K, V>>>,
}

/// HashMap move iterator
pub struct HashMapMoveIterator<K, V> {
    priv iter: vec::MoveRevIterator<Option<Bucket<K, V>>>,
}

/// HashSet iterator
#[deriving(Clone)]
pub struct HashSetIterator<'self, K> {
    priv iter: vec::VecIterator<'self, Option<Bucket<K, ()>>>,
}

/// HashSet move iterator
pub struct HashSetMoveIterator<K> {
    priv iter: vec::MoveRevIterator<Option<Bucket<K, ()>>>,
}

impl<'self, K, V> Iterator<(&'self K, &'self V)> for HashMapIterator<'self, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'self K, &'self V)> {
        for elt in self.iter {
            match elt {
                &Some(ref bucket) => return Some((&bucket.key, &bucket.value)),
                &None => {},
            }
        }
        None
    }
}

impl<'self, K, V> Iterator<(&'self K, &'self mut V)> for HashMapMutIterator<'self, K, V> {
    #[inline]
    fn next(&mut self) -> Option<(&'self K, &'self mut V)> {
        for elt in self.iter {
            match elt {
                &Some(ref mut bucket) => return Some((&bucket.key, &mut bucket.value)),
                &None => {},
            }
        }
        None
    }
}

impl<K, V> Iterator<(K, V)> for HashMapMoveIterator<K, V> {
    #[inline]
    fn next(&mut self) -> Option<(K, V)> {
        for elt in self.iter {
            match elt {
                Some(Bucket {key, value, _}) => return Some((key, value)),
                None => {},
            }
        }
        None
    }
}

impl<'self, K> Iterator<&'self K> for HashSetIterator<'self, K> {
    #[inline]
    fn next(&mut self) -> Option<&'self K> {
        for elt in self.iter {
            match elt {
                &Some(ref bucket) => return Some(&bucket.key),
                &None => {},
            }
        }
        None
    }
}

impl<K> Iterator<K> for HashSetMoveIterator<K> {
    #[inline]
    fn next(&mut self) -> Option<K> {
        for elt in self.iter {
            match elt {
                Some(bucket) => return Some(bucket.key),
                None => {},
            }
        }
        None
    }
}

impl<K: Eq + Hash, V> FromIterator<(K, V)> for HashMap<K, V> {
    fn from_iterator<T: Iterator<(K, V)>>(iter: &mut T) -> HashMap<K, V> {
        let (lower, _) = iter.size_hint();
        let mut map = HashMap::with_capacity(lower);
        map.extend(iter);
        map
    }
}

impl<K: Eq + Hash, V> Extendable<(K, V)> for HashMap<K, V> {
    fn extend<T: Iterator<(K, V)>>(&mut self, iter: &mut T) {
        for (k, v) in *iter {
            self.insert(k, v);
        }
    }
}

/// An implementation of a hash set using the underlying representation of a
/// HashMap where the value is (). As with the `HashMap` type, a `HashSet`
/// requires that the elements implement the `Eq` and `Hash` traits.
pub struct HashSet<T> {
    priv map: HashMap<T, ()>
}

impl<T:Hash + Eq> Eq for HashSet<T> {
    fn eq(&self, other: &HashSet<T>) -> bool { self.map == other.map }
    fn ne(&self, other: &HashSet<T>) -> bool { self.map != other.map }
}

impl<T:Hash + Eq> Container for HashSet<T> {
    /// Return the number of elements in the set
    fn len(&self) -> uint { self.map.len() }
}

impl<T:Hash + Eq> Mutable for HashSet<T> {
    /// Clear the set, removing all values.
    fn clear(&mut self) { self.map.clear() }
}

impl<T:Hash + Eq> Set<T> for HashSet<T> {
    /// Return true if the set contains a value
    fn contains(&self, value: &T) -> bool { self.map.contains_key(value) }

    /// Return true if the set has no elements in common with `other`.
    /// This is equivalent to checking for an empty intersection.
    fn is_disjoint(&self, other: &HashSet<T>) -> bool {
        self.iter().all(|v| !other.contains(v))
    }

    /// Return true if the set is a subset of another
    fn is_subset(&self, other: &HashSet<T>) -> bool {
        self.iter().all(|v| other.contains(v))
    }

    /// Return true if the set is a superset of another
    fn is_superset(&self, other: &HashSet<T>) -> bool {
        other.is_subset(self)
    }
}

impl<T:Hash + Eq> MutableSet<T> for HashSet<T> {
    /// Add a value to the set. Return true if the value was not already
    /// present in the set.
    fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

    /// Remove a value from the set. Return true if the value was
    /// present in the set.
    fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
}

impl<T:Hash + Eq> HashSet<T> {
    /// Create an empty HashSet
    pub fn new() -> HashSet<T> {
        HashSet::with_capacity(INITIAL_CAPACITY)
    }

    /// Create an empty HashSet with space for at least `n` elements in
    /// the hash table.
    pub fn with_capacity(capacity: uint) -> HashSet<T> {
        HashSet { map: HashMap::with_capacity(capacity) }
    }

    /// Reserve space for at least `n` elements in the hash table.
    pub fn reserve_at_least(&mut self, n: uint) {
        self.map.reserve_at_least(n)
    }

    /// Returns true if the hash set contains a value equivalent to the
    /// given query value.
    pub fn contains_equiv<Q:Hash + Equiv<T>>(&self, value: &Q) -> bool {
      self.map.contains_key_equiv(value)
    }

    /// An iterator visiting all elements in arbitrary order.
    /// Iterator element type is &'a T.
    pub fn iter<'a>(&'a self) -> HashSetIterator<'a, T> {
        HashSetIterator { iter: self.map.buckets.iter() }
    }

    /// Creates a consuming iterator, that is, one that moves each value out
    /// of the set in arbitrary order. The set cannot be used after calling
    /// this.
    pub fn move_iter(self) -> HashSetMoveIterator<T> {
        // `move_rev_iter` is more efficient than `move_iter` for vectors
        HashSetMoveIterator {iter: self.map.buckets.move_rev_iter()}
    }

    /// Visit the values representing the difference
    pub fn difference_iter<'a>(&'a self, other: &'a HashSet<T>) -> SetAlgebraIter<'a, T> {
        Repeat::new(other)
            .zip(self.iter())
            .filter_map(|(other, elt)| {
                if !other.contains(elt) { Some(elt) } else { None }
            })
    }

    /// Visit the values representing the symmetric difference
    pub fn symmetric_difference_iter<'a>(&'a self, other: &'a HashSet<T>)
        -> Chain<SetAlgebraIter<'a, T>, SetAlgebraIter<'a, T>> {
        self.difference_iter(other).chain(other.difference_iter(self))
    }

    /// Visit the values representing the intersection
    pub fn intersection_iter<'a>(&'a self, other: &'a HashSet<T>)
        -> SetAlgebraIter<'a, T> {
        Repeat::new(other)
            .zip(self.iter())
            .filter_map(|(other, elt)| {
                if other.contains(elt) { Some(elt) } else { None }
            })
    }

    /// Visit the values representing the union
    pub fn union_iter<'a>(&'a self, other: &'a HashSet<T>)
        -> Chain<HashSetIterator<'a, T>, SetAlgebraIter<'a, T>> {
        self.iter().chain(other.difference_iter(self))
    }

}

impl<T:Hash + Eq + Clone> Clone for HashSet<T> {
    fn clone(&self) -> HashSet<T> {
        HashSet {
            map: self.map.clone()
        }
    }
}

impl<K: Eq + Hash> FromIterator<K> for HashSet<K> {
    fn from_iterator<T: Iterator<K>>(iter: &mut T) -> HashSet<K> {
        let (lower, _) = iter.size_hint();
        let mut set = HashSet::with_capacity(lower);
        set.extend(iter);
        set
    }
}

impl<K: Eq + Hash> Extendable<K> for HashSet<K> {
    fn extend<T: Iterator<K>>(&mut self, iter: &mut T) {
        for k in *iter {
            self.insert(k);
        }
    }
}

// `Repeat` is used to feed the filter closure an explicit capture
// of a reference to the other set
/// Set operations iterator
pub type SetAlgebraIter<'self, T> =
    FilterMap<'static,(&'self HashSet<T>, &'self T), &'self T,
              Zip<Repeat<&'self HashSet<T>>,HashSetIterator<'self,T>>>;


#[cfg(test)]
mod test_map {
    use prelude::*;
    use super::*;

    #[test]
    fn test_create_capacity_zero() {
        let mut m = HashMap::with_capacity(0);
        assert!(m.insert(1, 1));
    }

    #[test]
    fn test_insert() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2));
        assert!(m.insert(2, 4));
        assert_eq!(*m.get(&1), 2);
        assert_eq!(*m.get(&2), 4);
    }

    #[test]
    fn test_find_mut() {
        let mut m = HashMap::new();
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
    fn test_insert_overwrite() {
        let mut m = HashMap::new();
        assert!(m.insert(1, 2));
        assert_eq!(*m.get(&1), 2);
        assert!(!m.insert(1, 3));
        assert_eq!(*m.get(&1), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert_eq!(*m.get(&9), 4);
        assert_eq!(*m.get(&5), 3);
        assert_eq!(*m.get(&1), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(m.insert(5, 3));
        assert!(m.insert(9, 4));
        assert!(m.remove(&1));
        assert_eq!(*m.get(&9), 4);
        assert_eq!(*m.get(&5), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = HashMap::with_capacity(4);
        assert!(m.insert(1, 2));
        assert!(!m.is_empty());
        assert!(m.remove(&1));
        assert!(m.is_empty());
    }

    #[test]
    fn test_pop() {
        let mut m = HashMap::new();
        m.insert(1, 2);
        assert_eq!(m.pop(&1), Some(2));
        assert_eq!(m.pop(&1), None);
    }

    #[test]
    fn test_swap() {
        let mut m = HashMap::new();
        assert_eq!(m.swap(1, 2), None);
        assert_eq!(m.swap(1, 3), Some(2));
        assert_eq!(m.swap(1, 4), Some(3));
    }

    #[test]
    fn test_find_or_insert() {
        let mut m: HashMap<int,int> = HashMap::new();
        assert_eq!(*m.find_or_insert(1, 2), 2);
        assert_eq!(*m.find_or_insert(1, 3), 2);
    }

    #[test]
    fn test_find_or_insert_with() {
        let mut m: HashMap<int,int> = HashMap::new();
        assert_eq!(*m.find_or_insert_with(1, |_| 2), 2);
        assert_eq!(*m.find_or_insert_with(1, |_| 3), 2);
    }

    #[test]
    fn test_insert_or_update_with() {
        let mut m: HashMap<int,int> = HashMap::new();
        assert_eq!(*m.insert_or_update_with(1, 2, |_,x| *x+=1), 2);
        assert_eq!(*m.insert_or_update_with(1, 2, |_,x| *x+=1), 3);
    }

    #[test]
    fn test_move_iter() {
        let hm = {
            let mut hm = HashMap::new();

            hm.insert('a', 1);
            hm.insert('b', 2);

            hm
        };

        let v = hm.move_iter().collect::<~[(char, int)]>();
        assert!([('a', 1), ('b', 2)] == v || [('b', 2), ('a', 1)] == v);
    }

    #[test]
    fn test_iterate() {
        let mut m = HashMap::with_capacity(4);
        for i in range(0u, 32) {
            assert!(m.insert(i, i*2));
        }
        let mut observed = 0;
        for (k, v) in m.iter() {
            assert_eq!(*v, *k * 2);
            observed |= (1 << *k);
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_find() {
        let mut m = HashMap::new();
        assert!(m.find(&1).is_none());
        m.insert(1, 2);
        match m.find(&1) {
            None => fail!(),
            Some(v) => assert!(*v == 2)
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = HashMap::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = HashMap::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_expand() {
        let mut m = HashMap::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0u;
        let old_resize_at = m.resize_at;
        while old_resize_at == m.resize_at {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_find_equiv() {
        let mut m = HashMap::new();

        let (foo, bar, baz) = (1,2,3);
        m.insert(~"foo", foo);
        m.insert(~"bar", bar);
        m.insert(~"baz", baz);


        assert_eq!(m.find_equiv(&("foo")), Some(&foo));
        assert_eq!(m.find_equiv(&("bar")), Some(&bar));
        assert_eq!(m.find_equiv(&("baz")), Some(&baz));

        assert_eq!(m.find_equiv(&("qux")), None);
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: HashMap<int, int> = xs.iter().map(|&x| x).collect();

        for &(k, v) in xs.iter() {
            assert_eq!(map.find(&k), Some(&v));
        }
    }
}

#[cfg(test)]
mod test_set {
    use super::*;
    use prelude::*;
    use container::Container;
    use vec::ImmutableEqVector;

    #[test]
    fn test_disjoint() {
        let mut xs = HashSet::new();
        let mut ys = HashSet::new();
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
        assert!(ys.insert(-11));
        assert!(xs.is_disjoint(&ys));
        assert!(ys.is_disjoint(&xs));
        assert!(ys.insert(7));
        assert!(!xs.is_disjoint(&ys));
        assert!(!ys.is_disjoint(&xs));
    }

    #[test]
    fn test_subset_and_superset() {
        let mut a = HashSet::new();
        assert!(a.insert(0));
        assert!(a.insert(5));
        assert!(a.insert(11));
        assert!(a.insert(7));

        let mut b = HashSet::new();
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
    fn test_iterate() {
        let mut a = HashSet::new();
        for i in range(0u, 32) {
            assert!(a.insert(i));
        }
        let mut observed = 0;
        for k in a.iter() {
            observed |= (1 << *k);
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_intersection() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(11));
        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(77));
        assert!(a.insert(103));
        assert!(a.insert(5));
        assert!(a.insert(-5));

        assert!(b.insert(2));
        assert!(b.insert(11));
        assert!(b.insert(77));
        assert!(b.insert(-9));
        assert!(b.insert(-42));
        assert!(b.insert(5));
        assert!(b.insert(3));

        let mut i = 0;
        let expected = [3, 5, 11, 77];
        for x in a.intersection_iter(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(3));
        assert!(b.insert(9));

        let mut i = 0;
        let expected = [1, 5, 11];
        for x in a.difference_iter(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_symmetric_difference() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));

        assert!(b.insert(-2));
        assert!(b.insert(3));
        assert!(b.insert(9));
        assert!(b.insert(14));
        assert!(b.insert(22));

        let mut i = 0;
        let expected = [-2, 1, 5, 11, 14, 22];
        for x in a.symmetric_difference_iter(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_union() {
        let mut a = HashSet::new();
        let mut b = HashSet::new();

        assert!(a.insert(1));
        assert!(a.insert(3));
        assert!(a.insert(5));
        assert!(a.insert(9));
        assert!(a.insert(11));
        assert!(a.insert(16));
        assert!(a.insert(19));
        assert!(a.insert(24));

        assert!(b.insert(-2));
        assert!(b.insert(1));
        assert!(b.insert(5));
        assert!(b.insert(9));
        assert!(b.insert(13));
        assert!(b.insert(19));

        let mut i = 0;
        let expected = [-2, 1, 3, 5, 9, 11, 13, 16, 19, 24];
        for x in a.union_iter(&b) {
            assert!(expected.contains(x));
            i += 1
        }
        assert_eq!(i, expected.len());
    }

    #[test]
    fn test_from_iter() {
        let xs = ~[1, 2, 3, 4, 5, 6, 7, 8, 9];

        let set: HashSet<int> = xs.iter().map(|&x| x).collect();

        for x in xs.iter() {
            assert!(set.contains(x));
        }
    }

    #[test]
    fn test_move_iter() {
        let hs = {
            let mut hs = HashSet::new();

            hs.insert('a');
            hs.insert('b');

            hs
        };

        let v = hs.move_iter().collect::<~[char]>();
        assert!(['a', 'b'] == v || ['b', 'a'] == v);
    }

    #[test]
    fn test_eq() {
        let mut s1 = HashSet::new();
        s1.insert(1);
        s1.insert(2);
        s1.insert(3);

        let mut s2 = HashSet::new();
        s2.insert(1);
        s2.insert(2);

        assert!(s1 != s2);

        s2.insert(3);

        assert_eq!(s1, s2);
    }
}
