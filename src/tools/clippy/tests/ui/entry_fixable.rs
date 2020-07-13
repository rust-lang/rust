// run-rustfix

#![allow(unused, clippy::needless_pass_by_value)]
#![warn(clippy::map_entry)]

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

fn foo() {}

fn insert_if_absent0<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        m.insert(k, v);
    }
}

fn main() {}
