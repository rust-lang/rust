#![feature(test)]

extern crate test;

use std::hash::Hash;
use test::{black_box, Bencher};

use hashbrown::HashMap;
//use rustc_hash::FxHashMap as HashMap;
//use std::collections::HashMap;

fn new_map<K: Eq + Hash, V>() -> HashMap<K, V> {
    HashMap::default()
}

#[bench]
fn insert_i32(b: &mut Bencher) {
    b.iter(|| {
        let mut m: HashMap<i32, i32> = new_map();
        for i in 1..1001 {
            m.insert(i, i);
        }
        black_box(m);
    })
}

#[bench]
fn insert_i64(b: &mut Bencher) {
    b.iter(|| {
        let mut m: HashMap<i64, i64> = new_map();
        for i in 1..1001 {
            m.insert(i, i);
        }
        black_box(m);
    })
}

#[bench]
fn insert_erase_i32(b: &mut Bencher) {
    b.iter(|| {
        let mut m: HashMap<i32, i32> = new_map();
        for i in 1..1001 {
            m.insert(i, i);
        }
        black_box(&mut m);
        for i in 1..1001 {
            m.remove(&i);
        }
        black_box(m);
    })
}

#[bench]
fn insert_erase_i64(b: &mut Bencher) {
    b.iter(|| {
        let mut m: HashMap<i64, i64> = new_map();
        for i in 1..1001 {
            m.insert(i, i);
        }
        black_box(&mut m);
        for i in 1..1001 {
            m.remove(&i);
        }
        black_box(m);
    })
}

#[bench]
fn lookup_i32(b: &mut Bencher) {
    let mut m: HashMap<i32, i32> = new_map();
    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1..1001 {
            black_box(m.get(&i));
        }
    })
}

#[bench]
fn lookup_i64(b: &mut Bencher) {
    let mut m: HashMap<i64, i64> = new_map();
    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1..1001 {
            black_box(m.get(&i));
        }
    })
}

#[bench]
fn lookup_fail_i32(b: &mut Bencher) {
    let mut m: HashMap<i32, i32> = new_map();
    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1001..2001 {
            black_box(m.get(&i));
        }
    })
}

#[bench]
fn lookup_fail_i64(b: &mut Bencher) {
    let mut m: HashMap<i64, i64> = new_map();
    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in 1001..2001 {
            black_box(m.get(&i));
        }
    })
}

#[bench]
fn iter_i32(b: &mut Bencher) {
    let mut m: HashMap<i32, i32> = new_map();
    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in &m {
            black_box(i);
        }
    })
}

#[bench]
fn iter_i64(b: &mut Bencher) {
    let mut m: HashMap<i64, i64> = new_map();
    for i in 1..1001 {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in &m {
            black_box(i);
        }
    })
}
