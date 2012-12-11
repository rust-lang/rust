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

// error-pattern:1 == 2
extern mod std;
use comm::Chan;
use comm::Port;
use comm::recv;

fn child() { assert (1 == 2); }

fn parent() {
    let p = Port::<int>();
    task::spawn(|| child() );
    let x = recv(p);
}

// This task is not linked to the failure chain, but since the other
// tasks are going to fail the kernel, this one will fail too
fn sleeper() {
    let p = Port::<int>();
    let x = recv(p);
}

fn main() {
    task::spawn(|| sleeper() );
    task::spawn(|| parent() );
}