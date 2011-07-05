// FIXME all this stuff should be in the standard lib, and in fact is,
// but due to the way our snapshots currently work, rustc can't use it
// until after the next snapshot.

fn new_str_hash[V]() -> std::map::hashmap[str, V] {
    let std::map::hashfn[str] hasher = std::str::hash;
    let std::map::eqfn[str] eqer = std::str::eq;
    ret std::map::mk_hashmap[str, V](hasher, eqer);
}

fn new_int_hash[V]() -> std::map::hashmap[int, V] {
    fn hash_int(&int x) -> uint { ret x as uint; }
    fn eq_int(&int a, &int b) -> bool { ret a == b; }
    auto hasher = hash_int;
    auto eqer = eq_int;
    ret std::map::mk_hashmap[int, V](hasher, eqer);
}

fn new_uint_hash[V]() -> std::map::hashmap[uint, V] {
    fn hash_uint(&uint x) -> uint { ret x; }
    fn eq_uint(&uint a, &uint b) -> bool { ret a == b; }
    auto hasher = hash_uint;
    auto eqer = eq_uint;
    ret std::map::mk_hashmap[uint, V](hasher, eqer);
}

fn istr(int i) -> str { ret std::int::to_str(i, 10u); }

fn uistr(uint i) -> str { ret std::uint::to_str(i, 10u); }
