// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

struct X { x: (), }

impl Drop for X {
    fn drop(&self) {
        error!("destructor runs");
    }
}

struct Y { y: Option<X> }

fn main() {
    let x = Y { y: Some(X { x: () }) };
    match x.y {
        Some(_z) => { }, //~ ERROR cannot bind by-move when matching an lvalue
        None => fail!()
    }
}
