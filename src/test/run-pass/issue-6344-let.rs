// xfail-test #3874
// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
struct A { x: uint }

impl Drop for A {
    fn drop(&self) {}
}

fn main() {
    let a = A { x: 0 };

    let A { x: ref x } = a;
    info!("%?", x)
}
