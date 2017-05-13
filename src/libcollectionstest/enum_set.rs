// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

use collections::enum_set::{CLike, EnumSet};

use self::Foo::*;

#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(usize)]
enum Foo {
    A,
    B,
    C,
}

impl CLike for Foo {
    fn to_usize(&self) -> usize {
        *self as usize
    }

    fn from_usize(v: usize) -> Foo {
        unsafe { mem::transmute(v) }
    }
}

#[test]
fn test_new() {
    let e: EnumSet<Foo> = EnumSet::new();
    assert!(e.is_empty());
}

#[test]
fn test_show() {
    let mut e = EnumSet::new();
    assert!(format!("{:?}", e) == "{}");
    e.insert(A);
    assert!(format!("{:?}", e) == "{A}");
    e.insert(C);
    assert!(format!("{:?}", e) == "{A, C}");
}

#[test]
fn test_len() {
    let mut e = EnumSet::new();
    assert_eq!(e.len(), 0);
    e.insert(A);
    e.insert(B);
    e.insert(C);
    assert_eq!(e.len(), 3);
    e.remove(&A);
    assert_eq!(e.len(), 2);
    e.clear();
    assert_eq!(e.len(), 0);
}

///////////////////////////////////////////////////////////////////////////
// intersect

#[test]
fn test_two_empties_do_not_intersect() {
    let e1: EnumSet<Foo> = EnumSet::new();
    let e2: EnumSet<Foo> = EnumSet::new();
    assert!(e1.is_disjoint(&e2));
}

#[test]
fn test_empty_does_not_intersect_with_full() {
    let e1: EnumSet<Foo> = EnumSet::new();

    let mut e2: EnumSet<Foo> = EnumSet::new();
    e2.insert(A);
    e2.insert(B);
    e2.insert(C);

    assert!(e1.is_disjoint(&e2));
}

#[test]
fn test_disjoint_intersects() {
    let mut e1: EnumSet<Foo> = EnumSet::new();
    e1.insert(A);

    let mut e2: EnumSet<Foo> = EnumSet::new();
    e2.insert(B);

    assert!(e1.is_disjoint(&e2));
}

#[test]
fn test_overlapping_intersects() {
    let mut e1: EnumSet<Foo> = EnumSet::new();
    e1.insert(A);

    let mut e2: EnumSet<Foo> = EnumSet::new();
    e2.insert(A);
    e2.insert(B);

    assert!(!e1.is_disjoint(&e2));
}

///////////////////////////////////////////////////////////////////////////
// contains and contains_elem

#[test]
fn test_superset() {
    let mut e1: EnumSet<Foo> = EnumSet::new();
    e1.insert(A);

    let mut e2: EnumSet<Foo> = EnumSet::new();
    e2.insert(A);
    e2.insert(B);

    let mut e3: EnumSet<Foo> = EnumSet::new();
    e3.insert(C);

    assert!(e1.is_subset(&e2));
    assert!(e2.is_superset(&e1));
    assert!(!e3.is_superset(&e2));
    assert!(!e2.is_superset(&e3))
}

#[test]
fn test_contains() {
    let mut e1: EnumSet<Foo> = EnumSet::new();
    e1.insert(A);
    assert!(e1.contains(&A));
    assert!(!e1.contains(&B));
    assert!(!e1.contains(&C));

    e1.insert(A);
    e1.insert(B);
    assert!(e1.contains(&A));
    assert!(e1.contains(&B));
    assert!(!e1.contains(&C));
}

///////////////////////////////////////////////////////////////////////////
// iter

#[test]
fn test_iterator() {
    let mut e1: EnumSet<Foo> = EnumSet::new();

    let elems: Vec<Foo> = e1.iter().collect();
    assert!(elems.is_empty());

    e1.insert(A);
    let elems: Vec<_> = e1.iter().collect();
    assert_eq!(elems, [A]);

    e1.insert(C);
    let elems: Vec<_> = e1.iter().collect();
    assert_eq!(elems, [A, C]);

    e1.insert(C);
    let elems: Vec<_> = e1.iter().collect();
    assert_eq!(elems, [A, C]);

    e1.insert(B);
    let elems: Vec<_> = e1.iter().collect();
    assert_eq!(elems, [A, B, C]);
}

///////////////////////////////////////////////////////////////////////////
// operators

#[test]
fn test_operators() {
    let mut e1: EnumSet<Foo> = EnumSet::new();
    e1.insert(A);
    e1.insert(C);

    let mut e2: EnumSet<Foo> = EnumSet::new();
    e2.insert(B);
    e2.insert(C);

    let e_union = e1 | e2;
    let elems: Vec<_> = e_union.iter().collect();
    assert_eq!(elems, [A, B, C]);

    let e_intersection = e1 & e2;
    let elems: Vec<_> = e_intersection.iter().collect();
    assert_eq!(elems, [C]);

    // Another way to express intersection
    let e_intersection = e1 - (e1 - e2);
    let elems: Vec<_> = e_intersection.iter().collect();
    assert_eq!(elems, [C]);

    let e_subtract = e1 - e2;
    let elems: Vec<_> = e_subtract.iter().collect();
    assert_eq!(elems, [A]);

    // Bitwise XOR of two sets, aka symmetric difference
    let e_symmetric_diff = e1 ^ e2;
    let elems: Vec<_> = e_symmetric_diff.iter().collect();
    assert_eq!(elems, [A, B]);

    // Another way to express symmetric difference
    let e_symmetric_diff = (e1 - e2) | (e2 - e1);
    let elems: Vec<_> = e_symmetric_diff.iter().collect();
    assert_eq!(elems, [A, B]);

    // Yet another way to express symmetric difference
    let e_symmetric_diff = (e1 | e2) - (e1 & e2);
    let elems: Vec<_> = e_symmetric_diff.iter().collect();
    assert_eq!(elems, [A, B]);
}

#[test]
#[should_panic]
fn test_overflow() {
    #[allow(dead_code)]
    #[derive(Copy, Clone)]
    #[repr(usize)]
    enum Bar {
        V00, V01, V02, V03, V04, V05, V06, V07, V08, V09,
        V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
        V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
        V30, V31, V32, V33, V34, V35, V36, V37, V38, V39,
        V40, V41, V42, V43, V44, V45, V46, V47, V48, V49,
        V50, V51, V52, V53, V54, V55, V56, V57, V58, V59,
        V60, V61, V62, V63, V64, V65, V66, V67, V68, V69,
    }

    impl CLike for Bar {
        fn to_usize(&self) -> usize {
            *self as usize
        }

        fn from_usize(v: usize) -> Bar {
            unsafe { mem::transmute(v) }
        }
    }
    let mut set = EnumSet::new();
    set.insert(Bar::V64);
}

#[test]
fn test_extend_ref() {
    let mut a = EnumSet::new();
    a.insert(A);

    a.extend(&[A, C]);

    assert_eq!(a.len(), 2);
    assert!(a.contains(&A));
    assert!(a.contains(&C));

    let mut b = EnumSet::new();
    b.insert(B);

    a.extend(&b);

    assert_eq!(a.len(), 3);
    assert!(a.contains(&A));
    assert!(a.contains(&B));
    assert!(a.contains(&C));
}
