// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z internal-lints

#![feature(rustc_private)]

extern crate rustc_data_structures;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
//~^ WARNING Prefer FxHashMap over HashMap, it has better performance
//~^^ WARNING Prefer FxHashSet over HashSet, it has better performance

#[deny(default_hash_types)]
fn main() {
    let _map: HashMap<String, String> = HashMap::default();
    //~^ ERROR Prefer FxHashMap over HashMap, it has better performance
    //~^^ ERROR Prefer FxHashMap over HashMap, it has better performance
    let _set: HashSet<String> = HashSet::default();
    //~^ ERROR Prefer FxHashSet over HashSet, it has better performance
    //~^^ ERROR Prefer FxHashSet over HashSet, it has better performance

    // test that the lint doesn't also match the Fx variants themselves
    let _fx_map: FxHashMap<String, String> = FxHashMap::default();
    let _fx_set: FxHashSet<String> = FxHashSet::default();
}
