// run-rustfix

#![allow(unused, clippy::needless_pass_by_value, clippy::collapsible_if)]
#![warn(clippy::map_entry)]

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

macro_rules! m {
    ($e:expr) => {{ $e }};
}

fn foo() {}

fn hash_map<K: Eq + Hash + Copy, V: Copy>(m: &mut HashMap<K, V>, k: K, v: V, v2: V) {
    if !m.contains_key(&k) {
        m.insert(k, v);
    }

    if !m.contains_key(&k) {
        if true {
            m.insert(k, v);
        } else {
            m.insert(k, v2);
        }
    }

    if !m.contains_key(&k) {
        if true {
            m.insert(k, v)
        } else {
            m.insert(k, v2)
        };
    }

    if !m.contains_key(&k) {
        if true {
            m.insert(k, v);
        } else {
            m.insert(k, v2);
            return;
        }
    }

    if !m.contains_key(&k) {
        foo();
        m.insert(k, v);
    }

    if !m.contains_key(&k) {
        match 0 {
            1 if true => {
                m.insert(k, v);
            },
            _ => {
                m.insert(k, v2);
            },
        };
    }

    if !m.contains_key(&k) {
        match 0 {
            0 => {},
            1 => {
                m.insert(k, v);
            },
            _ => {
                m.insert(k, v2);
            },
        };
    }

    if !m.contains_key(&k) {
        foo();
        match 0 {
            0 if false => {
                m.insert(k, v);
            },
            1 => {
                foo();
                m.insert(k, v);
            },
            2 | 3 => {
                for _ in 0..2 {
                    foo();
                }
                if true {
                    m.insert(k, v);
                } else {
                    m.insert(k, v2);
                };
            },
            _ => {
                m.insert(k, v2);
            },
        }
    }

    if !m.contains_key(&m!(k)) {
        m.insert(m!(k), m!(v));
    }
}

fn btree_map<K: Eq + Ord + Copy, V: Copy>(m: &mut BTreeMap<K, V>, k: K, v: V, v2: V) {
    if !m.contains_key(&k) {
        m.insert(k, v);
        foo();
    }
}

fn main() {}
