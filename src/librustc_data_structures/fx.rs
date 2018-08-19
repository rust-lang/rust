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
use std::hash::Hash;

pub use rustc_hash::FxHashMap;
pub use rustc_hash::FxHashSet;
pub use rustc_hash::FxHasher;

#[allow(non_snake_case)]
pub fn FxHashMap<K: Hash + Eq, V>() -> FxHashMap<K, V> {
    HashMap::default()
}

#[allow(non_snake_case)]
pub fn FxHashSet<V: Hash + Eq>() -> FxHashSet<V> {
    HashSet::default()
}

