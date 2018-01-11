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

// This is an important corner case pointed out by Niko: one is
// allowed to initiate a shared borrow during a reservation, but it
// *must end* before the activation occurs.
//
// FIXME: for clarity, diagnostics for these cases might be better off
// if they specifically said "cannot activate mutable borrow of `x`"

#![allow(dead_code)]

fn read(_: &i32) { }

fn ok() {
    let mut x = 3;
    let y = &mut x;
    { let z = &x; read(z); }
    *y += 1;
}

fn not_ok() {
    let mut x = 3;
    let y = &mut x;
    let z = &x;
    *y += 1;
    //[lxl]~^  ERROR cannot borrow `x` as mutable because it is also borrowed as immutable
    //[nll]~^^ ERROR cannot borrow `x` as mutable because it is also borrowed as immutable
    read(z);
}

fn should_be_ok_with_nll() {
    let mut x = 3;
    let y = &mut x;
    let z = &x;
    read(z);
    *y += 1;
    //[lxl]~^  ERROR cannot borrow `x` as mutable because it is also borrowed as immutable
    // (okay with nll today)
}

fn should_also_eventually_be_ok_with_nll() {
    let mut x = 3;
    let y = &mut x;
    let _z = &x;
    *y += 1;
    //[lxl]~^  ERROR cannot borrow `x` as mutable because it is also borrowed as immutable
}

fn main() { }
