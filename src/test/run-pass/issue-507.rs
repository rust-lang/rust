// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/*
   This is a test case for Issue 507.

   https://github.com/graydon/rust/issues/507
*/

fn grandchild(c: core::comm::Chan<int>) { core::comm::send(c, 42); }

fn child(c: core::comm::Chan<int>) {
    task::spawn(|| grandchild(c) )
}

fn main() {
    let p = core::comm::Port();
    let ch = core::comm::Chan(&p);

    task::spawn(|| child(ch) );

    let x: int = core::comm::recv(p);

    log(debug, x);

    assert (x == 42);
}
