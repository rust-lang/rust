// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: lxl nll
//[lxl]compile-flags: -Z borrowck=mir -Z two-phase-borrows
//[nll]compile-flags: -Z borrowck=mir -Z two-phase-borrows -Z nll

// This is similar to two-phase-reservation-sharing-interference.rs
// in that it shows a reservation that overlaps with a shared borrow.
//
// However, it is also more immediately concerning because one would
// intutively think that if run-pass/borrowck/two-phase-baseline.rs
// works, then this *should* work too.
//
// As before, the current implementation is (probably) more
// conservative than is necessary.
//
// So this test is just making a note of the current behavior, with
// the caveat that in the future, the rules may be loosened, at which
// point this test might be thrown out.

fn main() {
    let mut v = vec![0, 1, 2];
    let shared = &v;

    v.push(shared.len());
    //[lxl]~^  ERROR cannot borrow `v` as mutable because it is also borrowed as immutable [E0502]
    //[nll]~^^ ERROR cannot borrow `v` as mutable because it is also borrowed as immutable [E0502]

    assert_eq!(v, [0, 1, 2, 3]);
}
