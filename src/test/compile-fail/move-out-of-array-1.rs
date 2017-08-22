// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that we cannot move out of a fixed-size array (especially
// when the element type has a destructor).


struct D { _x: u8 }

impl Drop for D { fn drop(&mut self) { } }

fn main() {
    fn d() -> D { D { _x: 0 } }

    let _d1 = foo([d(), d(), d(), d()], 1);
    let _d3 = foo([d(), d(), d(), d()], 3);
}

fn foo(a: [D; 4], i: usize) -> D {
    a[i] //~ ERROR cannot move out of type `[D; 4]`, a non-copy array
}
