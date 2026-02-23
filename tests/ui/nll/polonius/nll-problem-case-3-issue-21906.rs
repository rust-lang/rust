#![crate_type = "lib"]

// This is part of a collection of regression tests related to the NLL problem case 3 that was
// deferred from the implementation of the NLL RFC, and left to be implemented by polonius. They are
// from open issues, e.g. tagged fixed-by-polonius, to ensure that the polonius alpha analysis does
// handle them, as does the datalog implementation.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #21906
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

use std::collections::HashMap;
use std::hash::Hash;

fn from_the_rfc<'r, K: Hash + Eq + Copy, V: Default>(
    map: &'r mut HashMap<K, V>,
    key: K,
) -> &'r mut V {
    match map.get_mut(&key) {
        Some(value) => value,
        None => {
            map.insert(key, V::default());
            map.get_mut(&key).unwrap()
        }
    }
}

// A variant that's similar to the RFC example above, but using the entry API, and requested in
// https://internals.rust-lang.org/t/get-mut-map-back-from-entry-api/24003
fn get_priority_mut_entry<'a, K, V>(
    map: &'a mut HashMap<K, V>,
    key1: K,
    key2: K,
) -> Option<&'a mut V>
where
    K: Eq + Hash,
{
    use std::collections::hash_map::Entry;
    match map.entry(key1) {
        Entry::Occupied(occupied) => Some(occupied.into_mut()),
        Entry::Vacant(_vacant) => match map.entry(key2) {
            Entry::Occupied(occupied2) => Some(occupied2.into_mut()),
            Entry::Vacant(_) => None,
        },
    }
}

// MCVE 1 from issue #21906
struct A {
    a: i32,
}

impl A {
    fn one(&mut self) -> &i32 {
        self.a = 10;
        &self.a
    }
    fn two(&mut self) -> &i32 {
        loop {
            let k = self.one();
            if *k > 10i32 {
                return k;
            }
        }
    }
}

// MCVE 2
struct Foo {
    data: Option<i32>,
}

fn foo(x: &mut Foo) -> Option<&mut i32> {
    if let Some(y) = x.data.as_mut() {
        return Some(y);
    }

    println!("{:?}", x.data);
    None
}

fn mcve2() {
    let mut x = Foo { data: Some(1) };
    foo(&mut x);
}

// MCVE 3
fn f(vec: &mut Vec<u8>) -> &u8 {
    if let Some(n) = vec.iter_mut().find(|n| **n == 1) {
        *n = 10;
        n
    } else {
        vec.push(10);
        vec.last().unwrap()
    }
}

fn mcve3() {
    let mut vec = vec![1, 2, 3];
    f(&mut vec);
}
