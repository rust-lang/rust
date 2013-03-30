// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

extern mod std;
use std::oldmap::{map, hashmap, int_hash};

class keys<K:Copy,V:Copy,M:Copy + map<K,V>>
    : iter::base_iter<K> {

    let map: M;

    new(map: M) {
        self.map = map;
    }

    fn each(blk: &fn(K) -> bool) { self.map.each(|k, _v| blk(k) ) }
    fn size_hint() -> Option<uint> { Some(self.map.size()) }
    fn eachi(blk: &fn(uint, K) -> bool) { iter::eachi(self, blk) }
}

pub fn main() {
    let m = int_hash();
    m.insert(1, 2);
    m.insert(3, 4);
    assert!(iter::to_vec(keys(m)) == ~[1, 3]);
}
