// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Sendable hash maps.  Very much a work in progress.

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::Eq;
use hash::Hash;
use prelude::*;
use to_bytes::IterBytes;

/// Open addressing with linear probing.
pub mod linear {
    use iter::BaseIter;
    use container::{Container, Mutable, Map, Set};
    use cmp::Eq;
    use cmp;
    use hash::Hash;
    use kinds::Copy;
    use option::{None, Option, Some};
    use option;
    use rand;
    use to_bytes::IterBytes;
    use uint;
    use vec;

    const INITIAL_CAPACITY: uint = 32u; // 2^5

    struct Bucket<K:Eq Hash,V> {
        hash: uint,
        key: K,
        value: V,
    }
    pub struct LinearMap<K:Eq Hash,V> {
        k0: u64,
        k1: u64,
        resize_at: uint,
        size: uint,
        buckets: ~[Option<Bucket<K,V>>],
    }

    // FIXME(#3148) -- we could rewrite found_entry
    // to have type option<&bucket<K,V>> which would be nifty
    // However, that won't work until #3148 is fixed
    enum SearchResult {
        FoundEntry(uint), FoundHole(uint), TableFull
    }

    fn resize_at(capacity: uint) -> uint {
        ((capacity as float) * 3. / 4.) as uint
    }

    pub fn LinearMap<K:Eq Hash,V>() -> LinearMap<K,V> {
        linear_map_with_capacity(INITIAL_CAPACITY)
    }

    pub fn linear_map_with_capacity<K:Eq Hash,V>(
        initial_capacity: uint) -> LinearMap<K,V> {
        let r = rand::Rng();
        linear_map_with_capacity_and_keys(r.gen_u64(), r.gen_u64(),
                                          initial_capacity)
    }

    fn linear_map_with_capacity_and_keys<K:Eq Hash,V> (
        k0: u64, k1: u64,
        initial_capacity: uint) -> LinearMap<K,V> {
        LinearMap {
            k0: k0, k1: k1,
            resize_at: resize_at(initial_capacity),
            size: 0,
            buckets: vec::from_fn(initial_capacity, |_i| None)
        }
    }

    priv impl<K:Hash IterBytes Eq, V> LinearMap<K,V> {
        #[inline(always)]
        pure fn to_bucket(&const self,
                          h: uint) -> uint {
            // FIXME(#3041) borrow a more sophisticated technique here from
            // Gecko, for example borrowing from Knuth, as Eich so
            // colorfully argues for here:
            // https://bugzilla.mozilla.org/show_bug.cgi?id=743107#c22
            h % self.buckets.len()
        }

        #[inline(always)]
        pure fn next_bucket(&const self,
                            idx: uint,
                            len_buckets: uint) -> uint {
            let n = (idx + 1) % len_buckets;
            unsafe{ // argh. log not considered pure.
                debug!("next_bucket(%?, %?) = %?", idx, len_buckets, n);
            }
            return n;
        }

        #[inline(always)]
        pure fn bucket_sequence(&const self,
                                hash: uint,
                                op: fn(uint) -> bool) -> uint {
            let start_idx = self.to_bucket(hash);
            let len_buckets = self.buckets.len();
            let mut idx = start_idx;
            loop {
                if !op(idx) {
                    return idx;
                }
                idx = self.next_bucket(idx, len_buckets);
                if idx == start_idx {
                    return start_idx;
                }
            }
        }

        #[inline(always)]
        pure fn bucket_for_key(&const self,
                               buckets: &[Option<Bucket<K,V>>],
                               k: &K) -> SearchResult {
            let hash = k.hash_keyed(self.k0, self.k1) as uint;
            self.bucket_for_key_with_hash(buckets, hash, k)
        }

        #[inline(always)]
        pure fn bucket_for_key_with_hash(&const self,
                                         buckets: &[Option<Bucket<K,V>>],
                                         hash: uint,
                                         k: &K) -> SearchResult {
            let _ = for self.bucket_sequence(hash) |i| {
                match buckets[i] {
                    Some(ref bkt) => if bkt.hash == hash && *k == bkt.key {
                        return FoundEntry(i);
                    },
                    None => return FoundHole(i)
                }
            };
            return TableFull;
        }

        /// Expands the capacity of the array and re-inserts each
        /// of the existing buckets.
        fn expand(&mut self) {
            let old_capacity = self.buckets.len();
            let new_capacity = old_capacity * 2;
            self.resize_at = ((new_capacity as float) * 3.0 / 4.0) as uint;

            let mut old_buckets = vec::from_fn(new_capacity, |_i| None);
            self.buckets <-> old_buckets;

            self.size = 0;
            for uint::range(0, old_capacity) |i| {
                let mut bucket = None;
                bucket <-> old_buckets[i];
                self.insert_opt_bucket(move bucket);
            }
        }

        fn insert_opt_bucket(&mut self, bucket: Option<Bucket<K,V>>) {
            match move bucket {
                Some(Bucket {hash: move hash,
                             key: move key,
                             value: move value}) => {
                    self.insert_internal(hash, move key, move value);
                }
                None => {}
            }
        }

        /// Inserts the key value pair into the buckets.
        /// Assumes that there will be a bucket.
        /// True if there was no previous entry with that key
        fn insert_internal(&mut self, hash: uint, k: K, v: V) -> bool {
            match self.bucket_for_key_with_hash(self.buckets, hash, &k) {
                TableFull => { fail ~"Internal logic error"; }
                FoundHole(idx) => {
                    debug!("insert fresh (%?->%?) at idx %?, hash %?",
                           k, v, idx, hash);
                    self.buckets[idx] = Some(Bucket {hash: hash,
                                                     key: move k,
                                                     value: move v});
                    self.size += 1;
                    true
                }
                FoundEntry(idx) => {
                    debug!("insert overwrite (%?->%?) at idx %?, hash %?",
                           k, v, idx, hash);
                    self.buckets[idx] = Some(Bucket {hash: hash,
                                                     key: move k,
                                                     value: move v});
                    false
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
            let mut idx = match self.bucket_for_key_with_hash(self.buckets,
                                                              hash, k) {
                TableFull | FoundHole(_) => return None,
                FoundEntry(idx) => idx
            };

            let len_buckets = self.buckets.len();
            let mut bucket = None;
            self.buckets[idx] <-> bucket;

            let value = match move bucket {
                None => None,
                Some(move bucket) => {
                    let Bucket { value: move value, _ } = move bucket;
                    Some(move value)
                },
            };

            idx = self.next_bucket(idx, len_buckets);
            while self.buckets[idx].is_some() {
                let mut bucket = None;
                bucket <-> self.buckets[idx];
                self.insert_opt_bucket(move bucket);
                idx = self.next_bucket(idx, len_buckets);
            }
            self.size -= 1;

            move value

        }

        fn search(&self,
                  hash: uint,
                  op: fn(x: &Option<Bucket<K,V>>) -> bool) {
            let _ = self.bucket_sequence(hash, |i| op(&self.buckets[i]));
        }
    }

    impl <K: Hash IterBytes Eq, V> LinearMap<K, V>: Container {
        pure fn len(&self) -> uint { self.size }
        pure fn is_empty(&self) -> bool { self.len() == 0 }
    }

    impl <K: Hash IterBytes Eq, V> LinearMap<K, V>: Mutable {
        fn clear(&mut self) {
            for uint::range(0, self.buckets.len()) |idx| {
                self.buckets[idx] = None;
            }
            self.size = 0;
        }
    }

    impl <K: Hash IterBytes Eq, V> LinearMap<K, V>: Map<K, V> {
        pure fn contains_key(&self, k: &K) -> bool {
            match self.bucket_for_key(self.buckets, k) {
                FoundEntry(_) => {true}
                TableFull | FoundHole(_) => {false}
            }
        }

        pure fn each(&self, blk: fn(k: &K, v: &V) -> bool) {
            for vec::each(self.buckets) |slot| {
                let mut broke = false;
                do slot.iter |bucket| {
                    if !blk(&bucket.key, &bucket.value) {
                        broke = true; // FIXME(#3064) just write "break;"
                    }
                }
                if broke { break; }
            }
        }

        pure fn each_key(&self, blk: fn(k: &K) -> bool) {
            self.each(|k, _v| blk(k))
        }

        pure fn each_value(&self, blk: fn(v: &V) -> bool) {
            self.each(|_k, v| blk(v))
        }

        pure fn find(&self, k: &K) -> Option<&self/V> {
            match self.bucket_for_key(self.buckets, k) {
                FoundEntry(idx) => {
                    match self.buckets[idx] {
                        Some(ref bkt) => {
                            // FIXME(#3148)---should be inferred
                            let bkt: &self/Bucket<K,V> = bkt;
                            Some(&bkt.value)
                        }
                        None => {
                            fail ~"LinearMap::find: internal logic error"
                        }
                    }
                }
                TableFull | FoundHole(_) => {
                    None
                }
            }
        }

        fn insert(&mut self, k: K, v: V) -> bool {
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
            self.insert_internal(hash, move k, move v)
        }

        fn remove(&mut self, k: &K) -> bool {
            match self.pop(k) {
                Some(_) => true,
                None => false,
            }
        }
    }

    impl<K:Hash IterBytes Eq,V> LinearMap<K,V> {
        fn pop(&mut self, k: &K) -> Option<V> {
            let hash = k.hash_keyed(self.k0, self.k1) as uint;
            self.pop_internal(hash, k)
        }

        fn swap(&mut self, k: K, v: V) -> Option<V> {
            // this could be faster.
            let hash = k.hash_keyed(self.k0, self.k1) as uint;
            let old_value = self.pop_internal(hash, &k);

            if self.size >= self.resize_at {
                // n.b.: We could also do this after searching, so
                // that we do not resize if this call to insert is
                // simply going to update a key in place.  My sense
                // though is that it's worse to have to search through
                // buckets to find the right spot twice than to just
                // resize in this corner case.
                self.expand();
            }

            self.insert_internal(hash, move k, move v);

            move old_value
        }

        fn consume(&mut self, f: fn(K, V)) {
            let mut buckets = ~[];
            self.buckets <-> buckets;
            self.size = 0;

            do vec::consume(move buckets) |_i, bucket| {
                match move bucket {
                    None => { },
                    Some(move bucket) => {
                        let Bucket {
                            key: move key,
                            value: move value,
                            _
                        } = move bucket;
                        f(move key, move value)
                    }
                }
            }
        }

        pure fn get(&self, k: &K) -> &self/V {
            match self.find(k) {
                Some(v) => v,
                None => fail fmt!("No entry found for key: %?", k),
            }
        }
    }

    impl<K:Hash IterBytes Eq, V: Copy> LinearMap<K,V> {
        pure fn find_copy(&const self, k: &K) -> Option<V> {
            match self.bucket_for_key(self.buckets, k) {
                FoundEntry(idx) => {
                    // FIXME (#3148): Once we rewrite found_entry, this
                    // failure case won't be necessary
                    match self.buckets[idx] {
                        Some(Bucket {value: copy value, _}) => {Some(value)}
                        None => fail ~"LinearMap::find: internal logic error"
                    }
                }
                TableFull | FoundHole(_) => {
                    None
                }
            }
        }
    }

    impl<K:Hash IterBytes Eq, V: Eq> LinearMap<K, V>: Eq {
        pure fn eq(&self, other: &LinearMap<K, V>) -> bool {
            if self.len() != other.len() { return false; }

            for self.each |key, value| {
                match other.find(key) {
                    None => return false,
                    Some(v) => if value != v { return false },
                }
            }

            return true;
        }

        pure fn ne(&self, other: &LinearMap<K, V>) -> bool {
            !self.eq(other)
        }
    }

    pub struct LinearSet<T: Hash IterBytes Eq> {
        priv map: LinearMap<T, ()>
    }

    impl <T: Hash IterBytes Eq> LinearSet<T>: BaseIter<T> {
        /// Visit all values in order
        pure fn each(&self, f: fn(&T) -> bool) { self.map.each_key(f) }
        pure fn size_hint(&self) -> Option<uint> { Some(self.len()) }
    }

    impl <T: Hash IterBytes Eq> LinearSet<T>: Eq {
        pure fn eq(&self, other: &LinearSet<T>) -> bool {
            self.map == other.map
        }
        pure fn ne(&self, other: &LinearSet<T>) -> bool {
            self.map != other.map
        }
    }

    impl <T: Hash IterBytes Eq> LinearSet<T>: Container {
        pure fn len(&self) -> uint { self.map.len() }
        pure fn is_empty(&self) -> bool { self.map.is_empty() }
    }

    impl <T: Hash IterBytes Eq> LinearSet<T>: Mutable {
        fn clear(&mut self) { self.map.clear() }
    }

    impl <T: Hash IterBytes Eq> LinearSet<T>: Set<T> {
        /// Return true if the set contains a value
        pure fn contains(&self, value: &T) -> bool {
            self.map.contains_key(value)
        }

        /// Add a value to the set. Return true if the value was not already
        /// present in the set.
        fn insert(&mut self, value: T) -> bool { self.map.insert(value, ()) }

        /// Remove a value from the set. Return true if the value was
        /// present in the set.
        fn remove(&mut self, value: &T) -> bool { self.map.remove(value) }
    }

    pub impl <T: Hash IterBytes Eq> LinearSet<T> {
        /// Create an empty LinearSet
        static fn new() -> LinearSet<T> { LinearSet{map: LinearMap()} }
    }
}

#[test]
pub mod test {
    use option::{None, Some};
    use hashmap::linear::LinearMap;
    use hashmap::linear;
    use uint;

    #[test]
    pub fn inserts() {
        let mut m = ~LinearMap();
        assert m.insert(1, 2);
        assert m.insert(2, 4);
        assert *m.get(&1) == 2;
        assert *m.get(&2) == 4;
    }

    #[test]
    pub fn overwrite() {
        let mut m = ~LinearMap();
        assert m.insert(1, 2);
        assert *m.get(&1) == 2;
        assert !m.insert(1, 3);
        assert *m.get(&1) == 3;
    }

    #[test]
    pub fn conflicts() {
        let mut m = linear::linear_map_with_capacity(4);
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 4);
        assert *m.get(&9) == 4;
        assert *m.get(&5) == 3;
        assert *m.get(&1) == 2;
    }

    #[test]
    pub fn conflict_remove() {
        let mut m = linear::linear_map_with_capacity(4);
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 4);
        assert m.remove(&1);
        assert *m.get(&9) == 4;
        assert *m.get(&5) == 3;
    }

    #[test]
    pub fn empty() {
        let mut m = linear::linear_map_with_capacity(4);
        assert m.insert(1, 2);
        assert !m.is_empty();
        assert m.remove(&1);
        assert m.is_empty();
    }

    #[test]
    pub fn pops() {
        let mut m = ~LinearMap();
        m.insert(1, 2);
        assert m.pop(&1) == Some(2);
        assert m.pop(&1) == None;
    }

    #[test]
    pub fn swaps() {
        let mut m = ~LinearMap();
        assert m.swap(1, 2) == None;
        assert m.swap(1, 3) == Some(2);
        assert m.swap(1, 4) == Some(3);
    }

    #[test]
    pub fn consumes() {
        let mut m = ~LinearMap();
        assert m.insert(1, 2);
        assert m.insert(2, 3);
        let mut m2 = ~LinearMap();
        do m.consume |k, v| {
            m2.insert(k, v);
        }
        assert m.len() == 0;
        assert m2.len() == 2;
        assert m2.find_copy(&1) == Some(2);
        assert m2.find_copy(&2) == Some(3);
    }

    #[test]
    pub fn iterate() {
        let mut m = linear::linear_map_with_capacity(4);
        for uint::range(0, 32) |i| {
            assert m.insert(i, i*2);
        }
        let mut observed = 0;
        for m.each |k, v| {
            assert *v == *k * 2;
            observed |= (1 << *k);
        }
        assert observed == 0xFFFF_FFFF;
    }

    #[test]
    pub fn find() {
        let mut m = ~LinearMap();
        assert m.find(&1).is_none();
        m.insert(1, 2);
        match m.find(&1) {
            None => fail,
            Some(v) => assert *v == 2
        }
    }

    #[test]
    pub fn test_eq() {
        let mut m1 = ~LinearMap();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = ~LinearMap();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert m1 != m2;

        m2.insert(3, 4);

        assert m1 == m2;
    }

    #[test]
    pub fn test_expand() {
        let mut m = ~LinearMap();

        assert m.len() == 0;
        assert m.is_empty();

        let mut i = 0u;
        let old_resize_at = m.resize_at;
        while old_resize_at == m.resize_at {
            m.insert(i, i);
            i += 1;
        }

        assert m.len() == i;
        assert !m.is_empty();
    }
}
