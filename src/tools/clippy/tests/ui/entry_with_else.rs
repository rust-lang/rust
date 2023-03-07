// run-rustfix

#![allow(unused, clippy::needless_pass_by_value, clippy::collapsible_if)]
#![warn(clippy::map_entry)]

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

macro_rules! m {
    ($e:expr) => {{ $e }};
}

fn foo() {}

fn insert_if_absent0<K: Eq + Hash + Copy, V: Copy>(m: &mut HashMap<K, V>, k: K, v: V, v2: V) {
    if !m.contains_key(&k) {
        m.insert(k, v);
    } else {
        m.insert(k, v2);
    }

    if m.contains_key(&k) {
        m.insert(k, v);
    } else {
        m.insert(k, v2);
    }

    if !m.contains_key(&k) {
        m.insert(k, v);
    } else {
        foo();
    }

    if !m.contains_key(&k) {
        foo();
    } else {
        m.insert(k, v);
    }

    if !m.contains_key(&k) {
        m.insert(k, v);
    } else {
        m.insert(k, v2);
    }

    if m.contains_key(&k) {
        if true { m.insert(k, v) } else { m.insert(k, v2) }
    } else {
        m.insert(k, v)
    };

    if m.contains_key(&k) {
        foo();
        m.insert(k, v)
    } else {
        None
    };
}

fn main() {}
