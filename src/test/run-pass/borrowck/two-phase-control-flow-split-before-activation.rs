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

#![cfg_attr(nll, feature(nll))]

fn main() {
    let mut a = 0;
    let mut b = 0;
    let p = if maybe() {
        &mut a
    } else {
        &mut b
    };
    use_(p);
}

fn maybe() -> bool { false }
fn use_<T>(_: T) { }
