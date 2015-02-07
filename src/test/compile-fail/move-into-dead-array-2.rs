// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that we cannot move into an uninitialized fixed-size array.

struct D { _x: u8 }

fn d() -> D { D { _x: 0 } }

fn main() {
    foo([d(), d(), d(), d()], 1);
    foo([d(), d(), d(), d()], 3);
}

fn foo(mut a: [D; 4], i: usize) {
    drop(a);
    a[i] = d(); //~ ERROR use of moved value: `a`
}
