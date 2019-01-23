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

fn insert_if_absent1<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        foo();
        m.insert(k, v);
    }
}

fn insert_if_absent2<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        m.insert(k, v)
    } else {
        None
    };
}

fn insert_if_present2<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if m.contains_key(&k) {
        None
    } else {
        m.insert(k, v)
    };
}

fn insert_if_absent3<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        foo();
        m.insert(k, v)
    } else {
        None
    };
}

fn insert_if_present3<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if m.contains_key(&k) {
        None
    } else {
        foo();
        m.insert(k, v)
    };
}

fn insert_in_btreemap<K: Ord, V>(m: &mut BTreeMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) {
        foo();
        m.insert(k, v)
    } else {
        None
    };
}

fn insert_other_if_absent<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, o: K, v: V) {
    if !m.contains_key(&k) {
        m.insert(o, v);
    }
}

fn main() {}
