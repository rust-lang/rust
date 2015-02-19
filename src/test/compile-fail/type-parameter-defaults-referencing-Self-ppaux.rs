// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a default that references `Self` which is then used in an
// object type. Issue #18956. In this case, the value is supplied by
// the user, but pretty-printing the type during the error message
// caused an ICE.

trait MyAdd<Rhs=Self> { fn add(&self, other: &Rhs) -> Self; }

impl MyAdd for i32 {
    fn add(&self, other: &i32) -> i32 { *self + *other }
}

fn main() {
    let x = 5;
    let y = x as MyAdd<i32>;
    //~^ ERROR as `MyAdd<i32>`
}
