// run-rustfix

#![allow(unused, clippy::needless_pass_by_value, clippy::collapsible_if)]
#![warn(clippy::map_entry)]

use std::arch::asm;
use std::collections::HashMap;
use std::hash::Hash;

macro_rules! m {
    ($e:expr) => {{ $e }};
}

macro_rules! insert {
    ($map:expr, $key:expr, $val:expr) => {
        $map.insert($key, $val)
    };
}

fn foo() {}

fn hash_map<K: Eq + Hash + Copy, V: Copy>(m: &mut HashMap<K, V>, m2: &mut HashMap<K, V>, k: K, k2: K, v: V, v2: V) {
    // or_insert(v)
    if !m.contains_key(&k) {
        m.insert(k, v);
    }

    // semicolon on insert, use or_insert_with(..)
    if !m.contains_key(&k) {
        if true {
            m.insert(k, v);
        } else {
            m.insert(k, v2);
        }
    }

    // semicolon on if, use or_insert_with(..)
    if !m.contains_key(&k) {
        if true {
            m.insert(k, v)
        } else {
            m.insert(k, v2)
        };
    }

    // early return, use if let
    if !m.contains_key(&k) {
        if true {
            m.insert(k, v);
        } else {
            m.insert(k, v2);
            return;
        }
    }

    // use or_insert_with(..)
    if !m.contains_key(&k) {
        foo();
        m.insert(k, v);
    }

    // semicolon on insert and match, use or_insert_with(..)
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

    // one branch doesn't insert, use if let
    if !m.contains_key(&k) {
        match 0 {
            0 => foo(),
            _ => {
                m.insert(k, v2);
            },
        };
    }

    // use or_insert_with
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

    // ok, insert in loop
    if !m.contains_key(&k) {
        for _ in 0..2 {
            m.insert(k, v);
        }
    }

    // macro_expansion test, use or_insert(..)
    if !m.contains_key(&m!(k)) {
        m.insert(m!(k), m!(v));
    }

    // ok, map used before insertion
    if !m.contains_key(&k) {
        let _ = m.len();
        m.insert(k, v);
    }

    // ok, inline asm
    if !m.contains_key(&k) {
        unsafe { asm!("nop") }
        m.insert(k, v);
    }

    // ok, different keys.
    if !m.contains_key(&k) {
        m.insert(k2, v);
    }

    // ok, different maps
    if !m.contains_key(&k) {
        m2.insert(k, v);
    }

    // ok, insert in macro
    if !m.contains_key(&k) {
        insert!(m, k, v);
    }

    // or_insert_with. Partial move of a local declared in the closure is ok.
    if !m.contains_key(&k) {
        let x = (String::new(), String::new());
        let _ = x.0;
        m.insert(k, v);
    }
}

fn main() {}
