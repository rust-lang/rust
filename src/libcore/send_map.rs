/*!

Sendable hash maps.  Very much a work in progress.

*/


/**
 * A function that returns a hash of a value
 *
 * The hash should concentrate entropy in the lower bits.
 */
type HashFn<K> = pure fn~(K) -> uint;
type EqFn<K> = pure fn~(K, K) -> bool;

/// Open addressing with linear probing.
mod linear {
    export LinearMap, linear_map, linear_map_with_capacity, public_methods;

    const initial_capacity: uint = 32u; // 2^5
    type Bucket<K,V> = {hash: uint, key: K, value: V};
    enum LinearMap<K,V> {
        LinearMap_({
            hashfn: pure fn~(x: &K) -> uint,
            eqfn: pure fn~(x: &K, y: &K) -> bool,
            resize_at: uint,
            size: uint,
            buckets: ~[option<Bucket<K,V>>]})
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

    fn linear_map<K,V>(
        +hashfn: pure fn~(x: &K) -> uint,
        +eqfn: pure fn~(x: &K, y: &K) -> bool) -> LinearMap<K,V> {

        linear_map_with_capacity(hashfn, eqfn, 32)
    }

    fn linear_map_with_capacity<K,V>(
        +hashfn: pure fn~(x: &K) -> uint,
        +eqfn: pure fn~(x: &K, y: &K) -> bool,
        initial_capacity: uint) -> LinearMap<K,V> {

        LinearMap_({
            hashfn: hashfn,
            eqfn: eqfn,
            resize_at: resize_at(initial_capacity),
            size: 0,
            buckets: vec::from_fn(initial_capacity, |_i| none)})
    }

    priv impl<K, V> &const LinearMap<K,V> {
        #[inline(always)]
        pure fn to_bucket(h: uint) -> uint {
            // FIXME(#3041) borrow a more sophisticated technique here from
            // Gecko, for example borrowing from Knuth, as Eich so
            // colorfully argues for here:
            // https://bugzilla.mozilla.org/show_bug.cgi?id=743107#c22
            h % self.buckets.len()
        }

        #[inline(always)]
        pure fn next_bucket(idx: uint, len_buckets: uint) -> uint {
            let n = (idx + 1) % len_buckets;
            unsafe{ // argh. log not considered pure.
                debug!{"next_bucket(%?, %?) = %?", idx, len_buckets, n};
            }
            return n;
        }

        #[inline(always)]
        pure fn bucket_sequence(hash: uint, op: fn(uint) -> bool) -> uint {
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
        pure fn bucket_for_key(
            buckets: &[option<Bucket<K,V>>],
            k: &K) -> SearchResult {

            let hash = self.hashfn(k);
            self.bucket_for_key_with_hash(buckets, hash, k)
        }

        #[inline(always)]
        pure fn bucket_for_key_with_hash(
            buckets: &[option<Bucket<K,V>>],
            hash: uint,
            k: &K) -> SearchResult {

            let _ = for self.bucket_sequence(hash) |i| {
                match buckets[i] {
                  some(bkt) => if bkt.hash == hash && self.eqfn(k, &bkt.key) {
                    return FoundEntry(i);
                  },
                  none => return FoundHole(i)
                }
            };
            return TableFull;
        }
    }

    priv impl<K,V> &mut LinearMap<K,V> {
        /// Expands the capacity of the array and re-inserts each
        /// of the existing buckets.
        fn expand() {
            let old_capacity = self.buckets.len();
            let new_capacity = old_capacity * 2;
            self.resize_at = ((new_capacity as float) * 3.0 / 4.0) as uint;

            let mut old_buckets = vec::from_fn(new_capacity, |_i| none);
            self.buckets <-> old_buckets;

            for uint::range(0, old_capacity) |i| {
                let mut bucket = none;
                bucket <-> old_buckets[i];
                if bucket.is_some() {
                    self.insert_bucket(bucket);
                }
            }
        }

        fn insert_bucket(+bucket: option<Bucket<K,V>>) {
            let {hash, key, value} <- option::unwrap(bucket);
            let _ = self.insert_internal(hash, key, value);
        }

        /// Inserts the key value pair into the buckets.
        /// Assumes that there will be a bucket.
        /// True if there was no previous entry with that key
        fn insert_internal(hash: uint, +k: K, +v: V) -> bool {
            match self.bucket_for_key_with_hash(self.buckets, hash, &k) {
              TableFull => {fail ~"Internal logic error";}
              FoundHole(idx) => {
                debug!{"insert fresh (%?->%?) at idx %?, hash %?",
                       k, v, idx, hash};
                self.buckets[idx] = some({hash: hash, key: k, value: v});
                self.size += 1;
                return true;
              }
              FoundEntry(idx) => {
                debug!{"insert overwrite (%?->%?) at idx %?, hash %?",
                       k, v, idx, hash};
                self.buckets[idx] = some({hash: hash, key: k, value: v});
                return false;
              }
            }
        }
    }

    impl<K,V> &mut LinearMap<K,V> {
        fn insert(+k: K, +v: V) -> bool {
            if self.size >= self.resize_at {
                // n.b.: We could also do this after searching, so
                // that we do not resize if this call to insert is
                // simply going to update a key in place.  My sense
                // though is that it's worse to have to search through
                // buckets to find the right spot twice than to just
                // resize in this corner case.
                self.expand();
            }

            let hash = self.hashfn(&k);
            self.insert_internal(hash, k, v)
        }

        fn remove(k: &K) -> bool {
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
            self.buckets[idx] = none;
            idx = self.next_bucket(idx, len_buckets);
            while self.buckets[idx].is_some() {
                let mut bucket = none;
                bucket <-> self.buckets[idx];
                self.insert_bucket(bucket);
                idx = self.next_bucket(idx, len_buckets);
            }
            self.size -= 1;
            return true;
        }

        fn clear() {
            for uint::range(0, self.buckets.len()) |idx| {
                self.buckets[idx] = none;
            }
            self.size = 0;
        }
    }

    priv impl<K,V> &LinearMap<K,V> {
        fn search(hash: uint, op: fn(x: &option<Bucket<K,V>>) -> bool) {
            let _ = self.bucket_sequence(hash, |i| op(&self.buckets[i]));
        }
    }

    impl<K,V> &const LinearMap<K,V> {
        pure fn len() -> uint {
            self.size
        }

        pure fn is_empty() -> bool {
            self.len() == 0
        }

        fn contains_key(k: &K) -> bool {
            match self.bucket_for_key(self.buckets, k) {
              FoundEntry(_) => {true}
              TableFull | FoundHole(_) => {false}
            }
        }
    }

    impl<K,V: copy> &const LinearMap<K,V> {
        fn find(k: &K) -> option<V> {
            match self.bucket_for_key(self.buckets, k) {
              FoundEntry(idx) => {
                match self.buckets[idx] {
                  some(bkt) => {some(copy bkt.value)}
                  // FIXME (#3148): Will be able to get rid of this when we
                  // redefine SearchResult
                  none      => fail ~"LinearMap::find: internal logic error"
                }
              }
              TableFull | FoundHole(_) => {
                none
              }
            }
        }

        fn get(k: &K) -> V {
            let value = self.find(k);
            if value.is_none() {
                fail fmt!{"No entry found for key: %?", k};
            }
            option::unwrap(value)
        }

    }

    impl<K,V> &LinearMap<K,V> {
        /*
        FIXME(#3148)--region inference fails to capture needed deps

        fn find_ref(k: &K) -> option<&self/V> {
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

        fn each_ref(blk: fn(k: &K, v: &V) -> bool) {
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
        fn each_key_ref(blk: fn(k: &K) -> bool) {
            self.each_ref(|k, _v| blk(k))
        }
        fn each_value_ref(blk: fn(v: &V) -> bool) {
            self.each_ref(|_k, v| blk(v))
        }
    }

    impl<K: copy, V: copy> &LinearMap<K,V> {
        fn each(blk: fn(+K,+V) -> bool) {
            self.each_ref(|k,v| blk(copy *k, copy *v));
        }
    }
    impl<K: copy, V> &LinearMap<K,V> {
        fn each_key(blk: fn(+K) -> bool) {
            self.each_key_ref(|k| blk(copy *k));
        }
    }
    impl<K, V: copy> &LinearMap<K,V> {
        fn each_value(blk: fn(+V) -> bool) {
            self.each_value_ref(|v| blk(copy *v));
        }
    }
}

#[test]
mod test {

    import linear::{LinearMap, linear_map};

    pure fn uint_hash(x: &uint) -> uint { *x }
    pure fn uint_eq(x: &uint, y: &uint) -> bool { *x == *y }

    fn int_linear_map<V>() -> LinearMap<uint,V> {
        return linear_map(uint_hash, uint_eq);
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
        let mut m = ~linear::linear_map_with_capacity(uint_hash, uint_eq, 4);
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 4);
        assert m.get(&9) == 4;
        assert m.get(&5) == 3;
        assert m.get(&1) == 2;
    }

    #[test]
    fn conflict_remove() {
        let mut m = ~linear::linear_map_with_capacity(uint_hash, uint_eq, 4);
        assert m.insert(1, 2);
        assert m.insert(5, 3);
        assert m.insert(9, 4);
        assert m.remove(&1);
        assert m.get(&9) == 4;
        assert m.get(&5) == 3;
    }

    #[test]
    fn empty() {
        let mut m = ~linear::linear_map_with_capacity(uint_hash, uint_eq, 4);
        assert m.insert(1, 2);
        assert !m.is_empty();
        assert m.remove(&1);
        assert m.is_empty();
    }

    #[test]
    fn iterate() {
        let mut m = linear::linear_map_with_capacity(uint_hash, uint_eq, 4);
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
