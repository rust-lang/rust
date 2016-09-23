// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::hash::{Hasher, Hash, BuildHasherDefault};

pub type FnvHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FnvHasher>>;
pub type FnvHashSet<V> = HashSet<V, BuildHasherDefault<FnvHasher>>;

#[allow(non_snake_case)]
pub fn FnvHashMap<K: Hash + Eq, V>() -> FnvHashMap<K, V> {
    HashMap::default()
}

#[allow(non_snake_case)]
pub fn FnvHashSet<V: Hash + Eq>() -> FnvHashSet<V> {
    HashSet::default()
}

/// A speedy hash algorithm for node ids and def ids. The hashmap in
/// libcollections by default uses SipHash which isn't quite as speedy as we
/// want. In the compiler we're not really worried about DOS attempts, so we
/// just default to a non-cryptographic hash.
///
/// This uses FNV hashing, as described here:
/// http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
pub struct FnvHasher(u64);

impl Default for FnvHasher {
    /// Creates a `FnvHasher`, with a 64-bit hex initial value.
    #[inline]
    fn default() -> FnvHasher {
        FnvHasher(0xcbf29ce484222325)
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;
        for byte in bytes {
            hash = hash ^ (*byte as u64);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        *self = FnvHasher(hash);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub fn hash<T: Hash>(v: &T) -> u64 {
    let mut state = FnvHasher::default();
    v.hash(&mut state);
    state.finish()
}
