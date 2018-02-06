// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: lxl nll g2p
//[lxl]compile-flags: -Z borrowck=mir -Z two-phase-borrows
//[nll]compile-flags: -Z borrowck=mir -Z two-phase-borrows -Z nll
//[g2p]compile-flags: -Z borrowck=mir -Z two-phase-borrows -Z nll -Z two-phase-beyond-autoref

#![feature(rustc_attrs)]

// This is a test checking that when we limit two-phase borrows to
// method receivers, we do not let other kinds of auto-ref to leak
// through.
//
// The g2p revision illustrates the "undesirable" behavior you would
// otherwise observe without limiting the phasing to autoref on method
// receivers (namely, that the test would pass).

fn bar(x: &mut u32) {
    foo(x, *x);
    //[lxl]~^ ERROR cannot use `*x` because it was mutably borrowed [E0503]
    //[nll]~^^ ERROR cannot use `*x` because it was mutably borrowed [E0503]
}

fn foo(x: &mut u32, y: u32) {
    *x += y;
}

#[rustc_error]
fn main() { //[g2p]~ ERROR compilation successful
    bar(&mut 5);
}
