#![feature(test)]

extern crate hashbrown;
extern crate rustc_hash;
extern crate test;

use std::hash::Hash;
use test::Bencher;

use hashbrown::HashMap;
//use rustc_hash::FxHashMap as HashMap;
//use std::collections::HashMap;

fn new_map<K: Eq + Hash, V>() -> HashMap<K, V> {
    HashMap::default()
}

#[bench]
fn new_drop(b: &mut Bencher) {
    b.iter(|| {
        let m: HashMap<i32, i32> = new_map();
        assert_eq!(m.len(), 0);
    })
}

#[bench]
fn new_insert_drop(b: &mut Bencher) {
    b.iter(|| {
        let mut m = new_map();
        m.insert(0, 0);
        assert_eq!(m.len(), 1);
    })
}

#[bench]
fn grow_by_insertion(b: &mut Bencher) {
    let mut m = new_map();

    for i in 1..1001 {
        m.insert(i, i);
    }

    let mut k = 1001;

    b.iter(|| {
        m.insert(k, k);
        k += 1;
    });
}

#[bench]
fn find_existing(b: &mut Bencher) {
    let mut m = new_map();

    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1..1001 {
            m.contains_key(&i);
        }
    });
}

#[bench]
fn find_nonexisting(b: &mut Bencher) {
    let mut m = new_map();

    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1001..2001 {
            m.contains_key(&i);
        }
    });
}

#[bench]
fn hashmap_as_queue(b: &mut Bencher) {
    let mut m = new_map();

    for i in 1..1001 {
        m.insert(i, i);
    }

    let mut k = 1;

    b.iter(|| {
        m.remove(&k);
        m.insert(k + 1000, k + 1000);
        k += 1;
    });
}

#[bench]
fn get_remove_insert(b: &mut Bencher) {
    let mut m = new_map();

    for i in 1..1001 {
        m.insert(i, i);
    }

    let mut k = 1;

    b.iter(|| {
        m.get(&(k + 400));
        m.get(&(k + 2000));
        m.remove(&k);
        m.insert(k + 1000, k + 1000);
        k += 1;
    })
}
