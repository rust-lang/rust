//@aux-build:proc_macros.rs
#![feature(rustc_private)]
#![warn(clippy::iter_over_hash_type)]
use std::collections::{HashMap, HashSet};

extern crate proc_macros;

// Ensure it also works via type aliases (this isn't really the Fx hasher but that does not matter).
type FxBuildHasher = std::collections::hash_map::RandomState;
type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
type FxHashSet<K> = HashSet<K, FxBuildHasher>;

fn main() {
    let mut hash_set = HashSet::<i32>::new();
    let mut hash_map = HashMap::<i32, i32>::new();
    let mut fx_hash_map = FxHashMap::<i32, i32>::default();
    let mut fx_hash_set = FxHashSet::<i32>::default();
    let vec = Vec::<i32>::new();

    // test hashset
    for x in &hash_set {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_set.iter() {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_set.clone() {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_set.drain() {
        //~^ iter_over_hash_type
        let _ = x;
    }

    // test hashmap
    for (x, y) in &hash_map {
        //~^ iter_over_hash_type
        let _ = (x, y);
    }
    for x in hash_map.keys() {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_map.values() {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_map.values_mut() {
        //~^ iter_over_hash_type
        *x += 1;
    }
    for x in hash_map.iter() {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_map.clone() {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in hash_map.drain() {
        //~^ iter_over_hash_type
        let _ = x;
    }

    // test type-aliased hashers
    for x in fx_hash_set {
        //~^ iter_over_hash_type
        let _ = x;
    }
    for x in fx_hash_map {
        //~^ iter_over_hash_type
        let _ = x;
    }

    // shouldn't fire
    for x in &vec {
        let _ = x;
    }
    for x in vec {
        let _ = x;
    }

    // should not lint, this comes from an external crate
    proc_macros::external! {
      for _ in HashMap::<i32, i32>::new() {}
    }
}
