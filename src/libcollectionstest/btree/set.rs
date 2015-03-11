// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::BTreeSet;
use std::hash::{SipHasher, self};

#[test]
fn test_clone_eq() {
  let mut m = BTreeSet::new();

  m.insert(1);
  m.insert(2);

  assert!(m.clone() == m);
}

#[test]
fn test_hash() {
  let mut x = BTreeSet::new();
  let mut y = BTreeSet::new();

  x.insert(1);
  x.insert(2);
  x.insert(3);

  y.insert(3);
  y.insert(2);
  y.insert(1);

  assert!(hash::hash::<_, SipHasher>(&x) == hash::hash::<_, SipHasher>(&y));
}

struct Counter<'a, 'b> {
    i: &'a mut usize,
    expected: &'b [i32],
}

impl<'a, 'b, 'c> FnMut<(&'c i32,)> for Counter<'a, 'b> {
    type Output = bool;

    extern "rust-call" fn call_mut(&mut self, (&x,): (&'c i32,)) -> bool {
        assert_eq!(x, self.expected[*self.i]);
        *self.i += 1;
        true
    }
}

fn check<F>(a: &[i32], b: &[i32], expected: &[i32], f: F) where
    // FIXME Replace Counter with `Box<FnMut(_) -> _>`
    F: FnOnce(&BTreeSet<i32>, &BTreeSet<i32>, Counter) -> bool,
{
    let mut set_a = BTreeSet::new();
    let mut set_b = BTreeSet::new();

    for x in a { assert!(set_a.insert(*x)) }
    for y in b { assert!(set_b.insert(*y)) }

    let mut i = 0;
    f(&set_a, &set_b, Counter { i: &mut i, expected: expected });
    assert_eq!(i, expected.len());
}

#[test]
fn test_intersection() {
    fn check_intersection(a: &[i32], b: &[i32], expected: &[i32]) {
        check(a, b, expected, |x, y, f| x.intersection(y).all(f))
    }

    check_intersection(&[], &[], &[]);
    check_intersection(&[1, 2, 3], &[], &[]);
    check_intersection(&[], &[1, 2, 3], &[]);
    check_intersection(&[2], &[1, 2, 3], &[2]);
    check_intersection(&[1, 2, 3], &[2], &[2]);
    check_intersection(&[11, 1, 3, 77, 103, 5, -5],
                       &[2, 11, 77, -9, -42, 5, 3],
                       &[3, 5, 11, 77]);
}

#[test]
fn test_difference() {
    fn check_difference(a: &[i32], b: &[i32], expected: &[i32]) {
        check(a, b, expected, |x, y, f| x.difference(y).all(f))
    }

    check_difference(&[], &[], &[]);
    check_difference(&[1, 12], &[], &[1, 12]);
    check_difference(&[], &[1, 2, 3, 9], &[]);
    check_difference(&[1, 3, 5, 9, 11],
                     &[3, 9],
                     &[1, 5, 11]);
    check_difference(&[-5, 11, 22, 33, 40, 42],
                     &[-12, -5, 14, 23, 34, 38, 39, 50],
                     &[11, 22, 33, 40, 42]);
}

#[test]
fn test_symmetric_difference() {
    fn check_symmetric_difference(a: &[i32], b: &[i32], expected: &[i32]) {
        check(a, b, expected, |x, y, f| x.symmetric_difference(y).all(f))
    }

    check_symmetric_difference(&[], &[], &[]);
    check_symmetric_difference(&[1, 2, 3], &[2], &[1, 3]);
    check_symmetric_difference(&[2], &[1, 2, 3], &[1, 3]);
    check_symmetric_difference(&[1, 3, 5, 9, 11],
                               &[-2, 3, 9, 14, 22],
                               &[-2, 1, 5, 11, 14, 22]);
}

#[test]
fn test_union() {
    fn check_union(a: &[i32], b: &[i32], expected: &[i32]) {
        check(a, b, expected, |x, y, f| x.union(y).all(f))
    }

    check_union(&[], &[], &[]);
    check_union(&[1, 2, 3], &[2], &[1, 2, 3]);
    check_union(&[2], &[1, 2, 3], &[1, 2, 3]);
    check_union(&[1, 3, 5, 9, 11, 16, 19, 24],
                &[-2, 1, 5, 9, 13, 19],
                &[-2, 1, 3, 5, 9, 11, 13, 16, 19, 24]);
}

#[test]
fn test_zip() {
    let mut x = BTreeSet::new();
    x.insert(5);
    x.insert(12);
    x.insert(11);

    let mut y = BTreeSet::new();
    y.insert("foo");
    y.insert("bar");

    let x = x;
    let y = y;
    let mut z = x.iter().zip(y.iter());

    // FIXME: #5801: this needs a type hint to compile...
    let result: Option<(&usize, & &'static str)> = z.next();
    assert_eq!(result.unwrap(), (&5, &("bar")));

    let result: Option<(&usize, & &'static str)> = z.next();
    assert_eq!(result.unwrap(), (&11, &("foo")));

    let result: Option<(&usize, & &'static str)> = z.next();
    assert!(result.is_none());
}

#[test]
fn test_from_iter() {
    let xs = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    let set: BTreeSet<_> = xs.iter().cloned().collect();

    for x in &xs {
        assert!(set.contains(x));
    }
}

#[test]
fn test_show() {
    let mut set = BTreeSet::new();
    let empty = BTreeSet::<i32>::new();

    set.insert(1);
    set.insert(2);

    let set_str = format!("{:?}", set);

    assert_eq!(set_str, "{1, 2}");
    assert_eq!(format!("{:?}", empty), "{}");
}
