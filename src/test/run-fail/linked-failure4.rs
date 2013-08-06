// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test linked failure
// error-pattern:1 == 2

use std::comm;
use std::task;

fn child() { assert!((1 == 2)); }

fn parent() {
    let (p, _c) = comm::stream::<int>();
    task::spawn(|| child() );
    let x = p.recv();
}

// This task is not linked to the failure chain, but since the other
// tasks are going to fail the kernel, this one will fail too
fn sleeper() {
    let (p, _c) = comm::stream::<int>();
    let x = p.recv();
}

fn main() {
    task::spawn(|| sleeper() );
    task::spawn(|| parent() );
}
