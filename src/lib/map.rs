/**
 * Hashmap implementation.
 */
type hashfn[K] = fn(&K) -> uint ;

type eqfn[K] = fn(&K, &K) -> bool ;

type hashmap[K, V] =
    obj {
        fn size() -> uint ;
        fn insert(&K, &V) -> bool ;
        fn contains_key(&K) -> bool ;
        fn get(&K) -> V ;
        fn find(&K) -> option::t[V] ;
        fn remove(&K) -> option::t[V] ;
        fn rehash() ;
        iter items() -> @rec(K key, V val);
        iter keys() -> K ;
    };
type hashset[K] = hashmap[K, ()];

fn set_add[K](hashset[K] set, &K key) -> bool {
    ret set.insert(key, ());
}

fn mk_hashmap[K, V](&hashfn[K] hasher, &eqfn[K] eqer) -> hashmap[K, V] {
    let uint initial_capacity = 32u; // 2^5

    let util::rational load_factor = rec(num=3, den=4);
    tag bucket[K, V] { nil; deleted; some(K, V); }
    fn make_buckets[K, V](uint nbkts) -> (bucket[K, V])[mutable] {
        ret ivec::init_elt_mut[bucket[K, V]](nil[K, V], nbkts);
    }
    // Derive two hash functions from the one given by taking the upper
    // half and lower half of the uint bits.  Our bucket probing
    // sequence is then defined by
    //
    //   hash(key, i) := hashl(key) * i + hashr(key)   for i = 0, 1, 2, ...
    //
    // Tearing the hash function apart this way is kosher in practice
    // as, assuming 32-bit uints, the table would have to be at 2^32
    // buckets before the resulting pair of hash functions no longer
    // probes all buckets for a fixed key.  Note that hashl is made to
    // output odd numbers (hence coprime to the number of nbkts, which
    // is always a power of 2), so that all buckets are probed for a
    // fixed key.

    fn hashl(uint n, uint nbkts) -> uint { ret (n >>> 16u) * 2u + 1u; }
    fn hashr(uint n, uint nbkts) -> uint { ret 0x0000_ffff_u & n; }
    fn hash(uint h, uint nbkts, uint i) -> uint {
        ret (hashl(h, nbkts) * i + hashr(h, nbkts)) % nbkts;
    }
    /**
     * We attempt to never call this with a full table.  If we do, it
     * will fail.
     */

    fn insert_common[K,
                     V](&hashfn[K] hasher, &eqfn[K] eqer,
                        &(bucket[K, V])[mutable] bkts, uint nbkts, &K key,
                        &V val) -> bool {
        let uint i = 0u;
        let uint h = hasher(key);
        while (i < nbkts) {
            let uint j = hash(h, nbkts, i);
            alt (bkts.(j)) {
                case (some(?k, _)) {
                    // Copy key to please alias analysis.

                    auto k_ = k;
                    if (eqer(key, k_)) {
                        bkts.(j) = some[K, V](k_, val);
                        ret false;
                    }
                    i += 1u;
                }
                case (_) { bkts.(j) = some[K, V](key, val); ret true; }
            }
        }
        fail; // full table

    }
    fn find_common[K,
                   V](&hashfn[K] hasher, &eqfn[K] eqer,
                      &(bucket[K, V])[mutable] bkts, uint nbkts, &K key) ->
       option::t[V] {
        let uint i = 0u;
        let uint h = hasher(key);
        while (i < nbkts) {
            let uint j = hash(h, nbkts, i);
            alt (bkts.(j)) {
                case (some(?k, ?v)) {
                    // Copy to please alias analysis.

                    auto k_ = k;
                    auto v_ = v;
                    if (eqer(key, k_)) { ret option::some[V](v_); }
                }
                case (nil) { ret option::none[V]; }
                case (deleted[K, V]) { }
            }
            i += 1u;
        }
        ret option::none[V];
    }
    fn rehash[K,
              V](&hashfn[K] hasher, &eqfn[K] eqer,
                 &(bucket[K, V])[mutable] oldbkts, uint noldbkts,
                 &(bucket[K, V])[mutable] newbkts, uint nnewbkts) {
        for (bucket[K, V] b in oldbkts) {
            alt (b) {
                case (some(?k_, ?v_)) {
                    auto k = k_;
                    auto v = v_;
                    insert_common[K,
                                  V](hasher, eqer, newbkts, nnewbkts, k, v);
                }
                case (_) { }
            }
        }
    }
    obj hashmap[K,
                V](hashfn[K] hasher,
                   eqfn[K] eqer,
                   mutable (bucket[K, V])[mutable] bkts,
                   mutable uint nbkts,
                   mutable uint nelts,
                   util::rational lf) {
        fn size() -> uint { ret nelts; }
        fn insert(&K key, &V val) -> bool {
            let util::rational load =
                rec(num=nelts + 1u as int, den=nbkts as int);
            if (!util::rational_leq(load, lf)) {
                let uint nnewbkts = uint::next_power_of_two(nbkts + 1u);
                auto newbkts = make_buckets[K, V](nnewbkts);
                rehash[K, V](hasher, eqer, bkts, nbkts, newbkts, nnewbkts);
                bkts = newbkts;
                nbkts = nnewbkts;
            }
            if (insert_common[K, V](hasher, eqer, bkts, nbkts, key, val)) {
                nelts += 1u;
                ret true;
            }
            ret false;
        }
        fn contains_key(&K key) -> bool {
            ret alt (find_common[K, V](hasher, eqer, bkts, nbkts, key)) {
                    case (option::some(_)) { true }
                    case (_) { false }
                };
        }
        fn get(&K key) -> V {
            ret alt (find_common[K, V](hasher, eqer, bkts, nbkts, key)) {
                    case (option::some(?val)) { val }
                    case (_) { fail }
                };
        }
        fn find(&K key) -> option::t[V] {
            be find_common[K, V](hasher, eqer, bkts, nbkts, key);
        }
        fn remove(&K key) -> option::t[V] {
            let uint i = 0u;
            let uint h = hasher(key);
            while (i < nbkts) {
                let uint j = hash(h, nbkts, i);
                alt (bkts.(j)) {
                    case (some(?k, ?v)) {
                        auto k_ = k;
                        auto vo = option::some(v);
                        if (eqer(key, k_)) {
                            bkts.(j) = deleted[K, V];
                            nelts -= 1u;
                            ret vo;
                        }
                    }
                    case (deleted) { }
                    case (nil) { ret option::none[V]; }
                }
                i += 1u;
            }
            ret option::none[V];
        }
        fn rehash() {
            auto newbkts = make_buckets[K, V](nbkts);
            rehash[K, V](hasher, eqer, bkts, nbkts, newbkts, nbkts);
            bkts = newbkts;
        }
        iter items() -> @rec(K key, V val) {
            for (bucket[K, V] b in bkts) {
                alt (b) {
                    case (some(?k, ?v)) { put @rec(key=k, val=v); }
                    case (_) { }
                }
            }
        }
        iter keys() -> K {
            for (bucket[K, V] b in bkts) {
                alt (b) {
                    case (some(?k, _)) { put k; }
                    case (_) { }
                }
            }
        }
    }
    auto bkts = make_buckets[K, V](initial_capacity);
    ret hashmap[K, V](hasher, eqer, bkts, initial_capacity, 0u, load_factor);
}

// Hash map constructors for basic types

fn new_str_hash[V]() -> hashmap[str, V] {
    ret mk_hashmap(str::hash, str::eq);
}

fn new_int_hash[V]() -> hashmap[int, V] {
    fn hash_int(&int x) -> uint { ret x as uint; }
    fn eq_int(&int a, &int b) -> bool { ret a == b; }
    ret mk_hashmap[int, V](hash_int, eq_int);
}

fn new_uint_hash[V]() -> hashmap[uint, V] {
    fn hash_uint(&uint x) -> uint { ret x; }
    fn eq_uint(&uint a, &uint b) -> bool { ret a == b; }
    ret mk_hashmap[uint, V](hash_uint, eq_uint);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
