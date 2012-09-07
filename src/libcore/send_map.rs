/*!

Sendable hash maps.  Very much a work in progress.

*/

use cmp::Eq;
use hash::Hash;
use to_bytes::IterBytes;

trait SendMap<K:Eq Hash, V: Copy> {
    // FIXME(#3148)  ^^^^ once find_ref() works, we can drop V:copy

    fn insert(&mut self, +k: K, +v: V) -> bool;
    fn remove(&mut self, k: &K) -> bool;
    fn clear(&mut self);
    pure fn len(&const self) -> uint;
    pure fn is_empty(&const self) -> bool;
    fn contains_key(&const self, k: &K) -> bool;
    fn each_ref(&self, blk: fn(k: &K, v: &V) -> bool);
    fn each_key_ref(&self, blk: fn(k: &K) -> bool);
    fn each_value_ref(&self, blk: fn(v: &V) -> bool);
    fn find(&const self, k: &K) -> Option<V>;
    fn get(&const self, k: &K) -> V;
}

/// Open addressing with linear probing.
mod linear {
    export LinearMap, linear_map, linear_map_with_capacity, public_methods;

    const initial_capacity: uint = 32u; // 2^5
    struct Bucket<K:Eq Hash,V> {
        hash: uint,
        key: K,
        value: V,
    }
    struct LinearMap<K:Eq Hash,V> {
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

    fn LinearMap<K:Eq Hash,V>() -> LinearMap<K,V> {
        linear_map_with_capacity(32)
    }

    fn linear_map_with_capacity<K:Eq Hash,V>(
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
                  Some(bkt) => if bkt.hash == hash && *k == bkt.key {
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

            for uint::range(0, old_capacity) |i| {
                let mut bucket = None;
                bucket <-> old_buckets[i];
                self.insert_opt_bucket(bucket);
            }
        }

        fn insert_opt_bucket(&mut self, +bucket: Option<Bucket<K,V>>) {
            match move bucket {
              Some(Bucket {hash: move hash,
                           key: move key,
                           value: move value}) => {
                self.insert_internal(hash, key, value);
              }
              None => {}
            }
        }

        /// Inserts the key value pair into the buckets.
        /// Assumes that there will be a bucket.
        /// True if there was no previous entry with that key
        fn insert_internal(&mut self, hash: uint, +k: K, +v: V) -> bool {
            match self.bucket_for_key_with_hash(self.buckets, hash, &k) {
              TableFull => {fail ~"Internal logic error";}
              FoundHole(idx) => {
                debug!("insert fresh (%?->%?) at idx %?, hash %?",
                       k, v, idx, hash);
                self.buckets[idx] = Some(Bucket {hash: hash,
                                                 key: k,
                                                 value: v});
                self.size += 1;
                return true;
              }
              FoundEntry(idx) => {
                debug!("insert overwrite (%?->%?) at idx %?, hash %?",
                       k, v, idx, hash);
                self.buckets[idx] = Some(Bucket {hash: hash,
                                                 key: k,
                                                 value: v});
                return false;
              }
            }
        }

        fn search(&self,
                  hash: uint,
                  op: fn(x: &Option<Bucket<K,V>>) -> bool) {
            let _ = self.bucket_sequence(hash, |i| op(&self.buckets[i]));
        }
    }

    impl<K:Hash IterBytes Eq,V> LinearMap<K,V> {
        fn insert(&mut self, +k: K, +v: V) -> bool {
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

        fn remove(&mut self, k: &K) -> bool {
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

            let mut idx = match self.bucket_for_key(self.buckets, k) {
              TableFull | FoundHole(_) => {
                return false;
              }
              FoundEntry(idx) => {
                idx
              }
            };

            let len_buckets = self.buckets.len();
            self.buckets[idx] = None;
            idx = self.next_bucket(idx, len_buckets);
            while self.buckets[idx].is_some() {
                let mut bucket = None;
                bucket <-> self.buckets[idx];
                self.insert_opt_bucket(bucket);
                idx = self.next_bucket(idx, len_buckets);
            }
            self.size -= 1;
            return true;
        }

        fn clear(&mut self) {
            for uint::range(0, self.buckets.len()) |idx| {
                self.buckets[idx] = None;
            }
            self.size = 0;
        }

        pure fn len(&const self) -> uint {
            self.size
        }

        pure fn is_empty(&const self) -> bool {
            self.len() == 0
        }

        fn contains_key(&const self,
                        k: &K) -> bool {
            match self.bucket_for_key(self.buckets, k) {
              FoundEntry(_) => {true}
              TableFull | FoundHole(_) => {false}
            }
        }

        /*
        FIXME(#3148)--region inference fails to capture needed deps

        fn find_ref(&self, k: &K) -> option<&self/V> {
            match self.bucket_for_key(self.buckets, k) {
              FoundEntry(idx) => {
                match check self.buckets[idx] {
                  some(ref bkt) => some(&bkt.value)
                }
              }
              TableFull | FoundHole(_) => {
                none
              }
            }
        }
        */

        fn each_ref(&self, blk: fn(k: &K, v: &V) -> bool) {
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

        fn each_key_ref(&self, blk: fn(k: &K) -> bool) {
            self.each_ref(|k, _v| blk(k))
        }

        fn each_value_ref(&self, blk: fn(v: &V) -> bool) {
            self.each_ref(|_k, v| blk(v))
        }
    }

    impl<K:Hash IterBytes Eq, V: Copy> LinearMap<K,V> {
        fn find(&const self, k: &K) -> Option<V> {
            match self.bucket_for_key(self.buckets, k) {
              FoundEntry(idx) => {
                // FIXME (#3148): Once we rewrite found_entry, this
                // failure case won't be necessary
                match self.buckets[idx] {
                    Some(bkt) => {Some(copy bkt.value)}
                    None => fail ~"LinearMap::find: internal logic error"
                }
              }
              TableFull | FoundHole(_) => {
                None
              }
            }
        }

        fn get(&const self, k: &K) -> V {
            let value = self.find(k);
            if value.is_none() {
                fail fmt!("No entry found for key: %?", k);
            }
            option::unwrap(value)
        }

    }

    impl<K: Hash IterBytes Eq Copy, V: Copy> LinearMap<K,V> {
        fn each(&self, blk: fn(+K,+V) -> bool) {
            self.each_ref(|k,v| blk(copy *k, copy *v));
        }
    }
    impl<K: Hash IterBytes Eq Copy, V> LinearMap<K,V> {
        fn each_key(&self, blk: fn(+K) -> bool) {
            self.each_key_ref(|k| blk(copy *k));
        }
    }
    impl<K: Hash IterBytes Eq, V: Copy> LinearMap<K,V> {
        fn each_value(&self, blk: fn(+V) -> bool) {
            self.each_value_ref(|v| blk(copy *v));
        }
    }
}

#[test]
mod test {

    import linear::LinearMap;

    fn int_linear_map<V>() -> LinearMap<uint,V> {
        return LinearMap();
    }

    #[test]
    fn inserts() {
        let mut m = ~int_linear_map();
        assert m.insert(1, 2);
        assert m.insert(2, 4);
        assert m.get(&1) == 2;
        assert m.get(&2) == 4;
    }

    #[test]
    fn overwrite() {
        let mut m = ~int_linear_map();
        assert m.insert(1, 2);
        assert m.get(&1) == 2;
        assert !m.insert(1, 3);
        assert m.get(&1) == 3;
    }

    #[test]
    fn conflicts() {
        let mut m = ~linear::linear_map_with_capacity(4);
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 4);
        assert m.get(&9) == 4;
        assert m.get(&5) == 3;
        assert m.get(&1) == 2;
    }

    #[test]
    fn conflict_remove() {
        let mut m = ~linear::linear_map_with_capacity(4);
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 4);
        assert m.remove(&1);
        assert m.get(&9) == 4;
        assert m.get(&5) == 3;
    }

    #[test]
    fn empty() {
        let mut m = ~linear::linear_map_with_capacity(4);
        assert m.insert(1, 2);
        assert !m.is_empty();
        assert m.remove(&1);
        assert m.is_empty();
    }

    #[test]
    fn iterate() {
        let mut m = linear::linear_map_with_capacity(4);
        for uint::range(0, 32) |i| {
            assert (&mut m).insert(i, i*2);
        }
        let mut observed = 0;
        for (&m).each |k, v| {
            assert v == k*2;
            observed |= (1 << k);
        }
        assert observed == 0xFFFF_FFFF;
    }
}
