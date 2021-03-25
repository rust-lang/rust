#![allow(unused, clippy::needless_pass_by_value)]
#![warn(clippy::map_entry)]

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

macro_rules! m {
    ($map:expr, $key:expr, $value:expr) => {
        $map.insert($key, $value)
    };
    ($e:expr) => {{ $e }};
}

fn foo() {}

// should not trigger
fn insert_other_if_absent<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, o: K, v: V) {
    if !m.contains_key(&k) {
        m.insert(o, v);
    }
}

// should not trigger, because the one uses different HashMap from another one
fn insert_from_different_map<K: Eq + Hash, V>(m: HashMap<K, V>, n: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        n.insert(k, v);
    }
}

// should not trigger, because the one uses different HashMap from another one
fn insert_from_different_map2<K: Eq + Hash, V>(m: &mut HashMap<K, V>, n: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        n.insert(k, v);
    }
}

fn insert_in_macro<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        m!(m, k, v);
    }
}

fn use_map_then_insert<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        let _ = m.len();
        m.insert(k, v);
    }
}

fn main() {}
