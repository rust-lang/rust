// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg(test)]

extern crate test;
use prelude::*;

use self::test::Bencher;
use iter::{range_inclusive};

#[bench]
fn new_drop(b : &mut Bencher) {
    use super::HashMap;

    b.iter(|| {
        let m : HashMap<int, int> = HashMap::new();
        assert_eq!(m.len(), 0);
    })
}

#[bench]
fn new_insert_drop(b : &mut Bencher) {
    use super::HashMap;

    b.iter(|| {
        let mut m = HashMap::new();
        m.insert(0i, 0i);
        assert_eq!(m.len(), 1);
    })
}

#[bench]
fn grow_by_insertion(b: &mut Bencher) {
    use super::HashMap;

    let mut m = HashMap::new();

    for i in range_inclusive(1i, 1000) {
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
    use super::HashMap;

    let mut m = HashMap::new();

    for i in range_inclusive(1i, 1000) {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in range_inclusive(1i, 1000) {
            m.contains_key(&i);
        }
    });
}

#[bench]
fn find_nonexisting(b: &mut Bencher) {
    use super::HashMap;

    let mut m = HashMap::new();

    for i in range_inclusive(1i, 1000) {
        m.insert(i, i);
    }

    b.iter(|| {
        for i in range_inclusive(1001i, 2000) {
            m.contains_key(&i);
        }
    });
}

#[bench]
fn hashmap_as_queue(b: &mut Bencher) {
    use super::HashMap;

    let mut m = HashMap::new();

    for i in range_inclusive(1i, 1000) {
        m.insert(i, i);
    }

    let mut k = 1i;

    b.iter(|| {
        m.pop(&k);
        m.insert(k + 1000, k + 1000);
        k += 1;
    });
}

#[bench]
fn find_pop_insert(b: &mut Bencher) {
    use super::HashMap;

    let mut m = HashMap::new();

    for i in range_inclusive(1i, 1000) {
        m.insert(i, i);
    }

    let mut k = 1i;

    b.iter(|| {
        m.find(&(k + 400));
        m.find(&(k + 2000));
        m.pop(&k);
        m.insert(k + 1000, k + 1000);
        k += 1;
    })
}
