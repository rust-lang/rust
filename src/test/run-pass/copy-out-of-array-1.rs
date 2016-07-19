// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that we can copy out of a fixed-size array.
//
// (Compare with compile-fail/move-out-of-array-1.rs)

#[derive(Copy, Clone)]
struct C { _x: u8 }

fn main() {
    fn d() -> C { C { _x: 0 } }

    let _d1 = foo([d(), d(), d(), d()], 1);
    let _d3 = foo([d(), d(), d(), d()], 3);
}

fn foo(a: [C; 4], i: usize) -> C {
    a[i]
}
