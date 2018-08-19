// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir -Z two-phase-borrows

// This is similar to two-phase-reservation-sharing-interference.rs
// in that it shows a reservation that overlaps with a shared borrow.
//
// Currently, this test fails with lexical lifetimes, but succeeds
// with non-lexical lifetimes. (The reason is because the activation
// of the mutable borrow ends up overlapping with a lexically-scoped
// shared borrow; but a non-lexical shared borrow can end before the
// activation occurs.)
//
// So this test is just making a note of the current behavior.

#![feature(rustc_attrs)]

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let mut v = vec![0, 1, 2];
    let shared = &v;

    v.push(shared.len());

    assert_eq!(v, [0, 1, 2, 3]);
}
