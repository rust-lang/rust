// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-17718.rs

extern crate "issue-17718" as other;

use std::sync::atomic;

const C1: uint = 1;
const C2: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;
const C3: fn() = foo;
const C4: uint = C1 * C1 + C1 / C1;
const C5: &'static uint = &C4;
const C6: uint = {
    const C: uint = 3;
    C
};

static S1: uint = 3;
static S2: atomic::AtomicUint = atomic::INIT_ATOMIC_UINT;

mod test {
    static A: uint = 4;
    static B: &'static uint = &A;
    static C: &'static uint = &(A);
}

fn foo() {}

fn main() {
    assert_eq!(C1, 1);
    assert_eq!(C3(), ());
    assert_eq!(C2.fetch_add(1, atomic::SeqCst), 0);
    assert_eq!(C2.fetch_add(1, atomic::SeqCst), 0);
    assert_eq!(C4, 2);
    assert_eq!(*C5, 2);
    assert_eq!(C6, 3);
    assert_eq!(S1, 3);
    assert_eq!(S2.fetch_add(1, atomic::SeqCst), 0);
    assert_eq!(S2.fetch_add(1, atomic::SeqCst), 1);

    match 1 {
        C1 => {}
        _ => unreachable!(),
    }

    let _a = C1;
    let _a = C2;
    let _a = C3;
    let _a = C4;
    let _a = C5;
    let _a = C6;
    let _a = S1;

    assert_eq!(other::C1, 1);
    assert_eq!(other::C3(), ());
    assert_eq!(other::C2.fetch_add(1, atomic::SeqCst), 0);
    assert_eq!(other::C2.fetch_add(1, atomic::SeqCst), 0);
    assert_eq!(other::C4, 2);
    assert_eq!(*other::C5, 2);
    assert_eq!(other::S1, 3);
    assert_eq!(other::S2.fetch_add(1, atomic::SeqCst), 0);
    assert_eq!(other::S2.fetch_add(1, atomic::SeqCst), 1);

    let _a = other::C1;
    let _a = other::C2;
    let _a = other::C3;
    let _a = other::C4;
    let _a = other::C5;

    match 1 {
        other::C1 => {}
        _ => unreachable!(),
    }
}
