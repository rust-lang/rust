//@aux-build:proc_macros.rs

#![warn(clippy::iter_over_hash_type)]
use std::collections::{HashMap, HashSet};

extern crate proc_macros;

fn main() {
    let mut hash_set = HashSet::<i32>::new();
    let mut hash_map = HashMap::<i32, i32>::new();
    let vec = Vec::<i32>::new();

    for x in &hash_set {
        let _ = x;
    }
    for x in hash_set.iter() {
        let _ = x;
    }
    for x in hash_set.clone() {
        let _ = x;
    }
    for x in hash_set.drain() {
        let _ = x;
    }
    for (x, y) in &hash_map {
        let _ = (x, y);
    }
    for x in hash_map.keys() {
        let _ = x;
    }
    for x in hash_map.values() {
        let _ = x;
    }
    for x in hash_map.values_mut() {
        *x += 1;
    }
    for x in hash_map.iter() {
        let _ = x;
    }
    for x in hash_map.clone() {
        let _ = x;
    }
    for x in hash_map.drain() {
        let _ = x;
    }

    // shouldnt fire
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
