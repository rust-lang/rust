// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn foo3() -> i32 {
    let mut x = 12;
    /// z //~ ERROR E0585
    while x < 1 {
        /// x //~ ERROR E0585
        //~^ ERROR attributes on non-item statements and expressions are experimental
        x += 1;
    }
    /// d //~ ERROR E0585
    return x;
}

fn main() {
    /// e //~ ERROR E0585
    foo3();
}
