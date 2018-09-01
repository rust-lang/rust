#![feature(tool_lints)]

#![warn(clippy::default_hash_types)]
#![feature(rustc_private)]

extern crate rustc_data_structures;

use std::collections::{HashMap, HashSet};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

fn main() {
    let _map: HashMap<String, String> = HashMap::default();
    let _set: HashSet<String> = HashSet::default();

    // test that the lint doesn't also match the Fx variants themselves 😂
    let _fx_map: FxHashMap<String, String> = FxHashMap::default();
    let _fx_set: FxHashSet<String> = FxHashSet::default();
}
