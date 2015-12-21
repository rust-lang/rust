#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused)]

#![deny(hashmap_entry)]

use std::collections::HashMap;
use std::hash::Hash;

fn insert_if_absent<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) { m.insert(k, v); } //~ERROR: usage of `contains_key` followed by `insert` on `HashMap`
}

fn insert_if_absent2<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, v: V) {
    if !m.contains_key(&k) { m.insert(k, v) } else { None }; //~ERROR: usage of `contains_key` followed by `insert` on `HashMap`
}

/* TODO
fn insert_other_if_absent<K: Eq + Hash, V>(m: &mut HashMap<K, V>, k: K, o: K, v: V) {
    if !m.contains_key(&k) { m.insert(o, v) } else { None };
}
*/

fn main() {
}
