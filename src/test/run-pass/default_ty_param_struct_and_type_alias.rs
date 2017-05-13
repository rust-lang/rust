// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#![feature(default_type_parameter_fallback)]

use std::marker::PhantomData;

pub struct DeterministicHasher;
pub struct RandomHasher;


pub struct MyHashMap<K, V, H=DeterministicHasher> {
    data: PhantomData<(K, V, H)>
}

impl<K, V, H> MyHashMap<K, V, H> {
    fn new() -> MyHashMap<K, V, H> {
        MyHashMap { data: PhantomData }
    }
}

mod mystd {
    use super::{MyHashMap, RandomHasher};
    pub type HashMap<K, V, H=RandomHasher> = MyHashMap<K, V, H>;
}

fn try_me<H>(hash_map: mystd::HashMap<i32, i32, H>) {}

fn main() {
    let hash_map = mystd::HashMap::new();
    try_me(hash_map);
}
