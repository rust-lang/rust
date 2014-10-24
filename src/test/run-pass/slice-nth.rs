// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate test;

use std::iter::count;
use std::slice::Items;
use test::Bencher;

// This wraps the Items struct, but uses the default version of nth()
struct ItemsWrapper<'a, T: 'a> {
    base: Items<'a, T>
}

impl<'a, T: 'a> Iterator<&'a T> for ItemsWrapper<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        self.base.next()
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.base.size_hint()
    }
}

pub fn main() {
    let vec1: Vec<uint> = count(0u, 1).take(100).collect();
    let vec2 = vec1.clone();

    let mut iter1 = vec1.iter();
    let mut iter2 = ItemsWrapper{ base: vec2.iter() };

    //Now, test that Items's implementation of nth() is equal to the
    //default Iterator implementation for a bunch of random offsets
    assert_eq!(iter1.nth(0), iter2.nth(0));
    assert_eq!(iter1.nth(1), iter2.nth(1));
    assert_eq!(iter1.next(), iter2.next());
    assert_eq!(iter1.nth(0), iter2.nth(0));
    assert_eq!(iter1.next(), iter2.next());
    assert_eq!(iter1.next(), iter2.next());
    assert_eq!(iter1.nth(2), iter2.nth(2));
    assert_eq!(iter1.nth(5), iter2.nth(5));
    assert_eq!(iter1.nth(5), iter2.nth(5));
    assert_eq!(iter1.next(), iter2.next());

    // Test that nth() hitting the last element leaves the iterator in a state
    // that next() will return None
    let vec1: Vec<uint> = count(0u, 1).take(2).collect();
    let vec2 = vec1.clone();

    let mut iter1 = vec1.iter();
    let mut iter2 = ItemsWrapper{ base: vec2.iter() };
    assert_eq!(iter1.nth(1), iter2.nth(1));
    assert!(iter1.next().is_none());
    assert!(iter2.next().is_none());

    // Test that n() returns None when the first element past the end is requested
    let mut iter1 = vec1.iter();
    let mut iter2 = ItemsWrapper{ base: vec2.iter() };
    assert_eq!(iter1.nth(1), iter2.nth(1));
    assert!(iter1.nth(0).is_none());
    assert!(iter2.nth(0).is_none());

    // Test that nth() returns None when elements past the end pointer are requested
    let mut iter1 = vec1.iter();
    let mut iter2 = ItemsWrapper{ base: vec2.iter() };
    assert_eq!(iter1.nth(1), iter2.nth(1));
    assert!(iter1.nth(1).is_none());
    assert!(iter2.nth(1).is_none());
}

#[bench]
fn bench_default_slice_nth(b: &mut Bencher) {
    let v = Vec::from_fn(10_000_000, |n| n);
    b.iter(|| {
        let mut iter = ItemsWrapper{ base: v.iter() };
        let mut sum = 0;
        loop {
            match iter.nth(1000) {
                None => break,
                Some(&x) => sum = sum + x,
            }
        }
        sum
    })
}

#[bench]
fn bench_slice_nth(b: &mut Bencher) {
    let v = Vec::from_fn(10_000_000, |n| n);
    b.iter(|| {
        let mut iter = v.iter();
        let mut sum = 0;
        loop {
            match iter.nth(1000) {
                None => break,
                Some(&x) => sum = sum + x,
            }
        }
        sum
    })
}
