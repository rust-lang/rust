// compile-flags: -Z unstable-options

#![feature(rustc_private)]

extern crate rustc_data_structures;

use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};

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
