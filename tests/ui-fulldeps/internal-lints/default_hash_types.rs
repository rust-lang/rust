//@ compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(rustc::default_hash_types)]

extern crate rustc_data_structures;

use rustc_data_structures::gx::{GxHashMap, GxHashSet};
use std::collections::{HashMap, HashSet};

mod foo {
    pub struct HashMap;
}

fn main() {
    let _map: HashMap<String, String> = HashMap::default();
    //~^ ERROR prefer `GxHashMap` over `HashMap`, it has better performance
    //~^^ ERROR prefer `GxHashMap` over `HashMap`, it has better performance
    let _set: HashSet<String> = HashSet::default();
    //~^ ERROR prefer `GxHashSet` over `HashSet`, it has better performance
    //~^^ ERROR prefer `GxHashSet` over `HashSet`, it has better performance

    // test that the lint doesn't also match the Fx variants themselves
    let _fx_map: GxHashMap<String, String> = GxHashMap::default();
    let _fx_set: GxHashSet<String> = GxHashSet::default();

    // test another struct of the same name
    let _ = foo::HashMap;
}
