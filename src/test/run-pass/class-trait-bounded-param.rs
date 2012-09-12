// xfail-test

extern mod std;
use std::map::{map, hashmap, int_hash};

class keys<K: Copy, V: Copy, M: Copy map<K,V>>
    : iter::base_iter<K> {

    let map: M;

    new(map: M) {
        self.map = map;
    }

    fn each(blk: fn(K) -> bool) { self.map.each(|k, _v| blk(k) ) }
    fn size_hint() -> option<uint> { some(self.map.size()) }
    fn eachi(blk: fn(uint, K) -> bool) { iter::eachi(self, blk) }
}

fn main() {
    let m = int_hash();
    m.insert(1, 2);
    m.insert(3, 4);
    assert iter::to_vec(keys(m)) == ~[1, 3];
}
