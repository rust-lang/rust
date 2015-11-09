// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

// check raw fat pointer ops in mir
// FIXME: please improve this when we get monomorphization support

use std::mem;

#[derive(Debug, PartialEq, Eq)]
struct ComparisonResults {
    lt: bool,
    le: bool,
    gt: bool,
    ge: bool,
    eq: bool,
    ne: bool
}

const LT: ComparisonResults = ComparisonResults {
    lt: true,
    le: true,
    gt: false,
    ge: false,
    eq: false,
    ne: true
};

const EQ: ComparisonResults = ComparisonResults {
    lt: false,
    le: true,
    gt: false,
    ge: true,
    eq: true,
    ne: false
};

const GT: ComparisonResults = ComparisonResults {
    lt: false,
    le: false,
    gt: true,
    ge: true,
    eq: false,
    ne: true
};

#[rustc_mir]
fn compare_su8(a: *const S<[u8]>, b: *const S<[u8]>) -> ComparisonResults {
    ComparisonResults {
        lt: a < b,
        le: a <= b,
        gt: a > b,
        ge: a >= b,
        eq: a == b,
        ne: a != b
    }
}

#[rustc_mir]
fn compare_au8(a: *const [u8], b: *const [u8]) -> ComparisonResults {
    ComparisonResults {
        lt: a < b,
        le: a <= b,
        gt: a > b,
        ge: a >= b,
        eq: a == b,
        ne: a != b
    }
}

#[rustc_mir(graphviz="comparefoo.gv")]
fn compare_foo<'a>(a: *const (Foo+'a), b: *const (Foo+'a)) -> ComparisonResults {
    ComparisonResults {
        lt: a < b,
        le: a <= b,
        gt: a > b,
        ge: a >= b,
        eq: a == b,
        ne: a != b
    }
}

#[rustc_mir(graphviz="simpleeq.gv")]
fn simple_eq<'a>(a: *const (Foo+'a), b: *const (Foo+'a)) -> bool {
    let result = a == b;
    result
}

fn assert_inorder<T: Copy>(a: &[T],
                           compare: fn(T, T) -> ComparisonResults) {
    for i in 0..a.len() {
        for j in 0..a.len() {
            let cres = compare(a[i], a[j]);
            if i < j {
                assert_eq!(cres, LT);
            } else if i == j {
                assert_eq!(cres, EQ);
            } else {
                assert_eq!(cres, GT);
            }
        }
    }
}

trait Foo { fn foo(&self) -> usize; }
impl<T> Foo for T {
    fn foo(&self) -> usize {
        mem::size_of::<T>()
    }
}

struct S<T:?Sized>(u32, T);

fn main() {
    let array = [0,1,2,3,4];
    let array2 = [5,6,7,8,9];

    // fat ptr comparison: addr then extra

    // check ordering for arrays
    let mut ptrs: Vec<*const [u8]> = vec![
        &array[0..0], &array[0..1], &array, &array[1..]
    ];

    let array_addr = &array as *const [u8] as *const u8 as usize;
    let array2_addr = &array2 as *const [u8] as *const u8 as usize;
    if array2_addr < array_addr {
        ptrs.insert(0, &array2);
    } else {
        ptrs.push(&array2);
    }
    assert_inorder(&ptrs, compare_au8);

    let u8_ = (0u8, 1u8);
    let u32_ = (4u32, 5u32);

    // check ordering for ptrs
    let buf: &mut [*const Foo] = &mut [
        &u8_, &u8_.0,
        &u32_, &u32_.0,
    ];
    buf.sort_by(|u,v| {
        let u : [*const (); 2] = unsafe { mem::transmute(*u) };
        let v : [*const (); 2] = unsafe { mem::transmute(*v) };
        u.cmp(&v)
    });
    assert_inorder(buf, compare_foo);

    // check ordering for structs containing arrays
    let ss: (S<[u8; 2]>,
             S<[u8; 3]>,
             S<[u8; 2]>) = (
        S(7, [8, 9]),
        S(10, [11, 12, 13]),
        S(4, [5, 6])
    );
    assert_inorder(&[
        &ss.0 as *const S<[u8]>,
        &ss.1 as *const S<[u8]>,
        &ss.2 as *const S<[u8]>
            ], compare_su8);

    assert!(simple_eq(&0u8 as *const _, &0u8 as *const _));
    assert!(!simple_eq(&0u8 as *const _, &1u8 as *const _));
}
