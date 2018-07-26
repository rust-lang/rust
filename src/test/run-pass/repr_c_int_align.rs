// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O

#![allow(dead_code)]

#[repr(C, u8)]
enum ReprCu8 {
    A(u16),
    B,
}

#[repr(u8)]
enum Repru8 {
    A(u16),
    B,
}

#[repr(C)]
struct ReprC {
    tag: u8,
    padding: u8,
    payload: u16,
}

fn main() {
    // Test `repr(C, u8)`.
    let r1 = ReprC { tag: 0, padding: 0, payload: 0 };
    let r2 = ReprC { tag: 0, padding: 1, payload: 1 };

    let t1: &ReprCu8 = unsafe { std::mem::transmute(&r1) };
    let t2: &ReprCu8 = unsafe { std::mem::transmute(&r2) };

    match (t1, t2) {
        (ReprCu8::A(_), ReprCu8::A(_)) => (),
        _ => assert!(false)
    };

    // Test `repr(u8)`.
    let t1: &Repru8 = unsafe { std::mem::transmute(&r1) };
    let t2: &Repru8 = unsafe { std::mem::transmute(&r2) };

    match (t1, t2) {
        (Repru8::A(_), Repru8::A(_)) => (),
        _ => assert!(false)
    };
}
