// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:crateresolve2-1.rs
// aux-build:crateresolve2-2.rs
// aux-build:crateresolve2-3.rs
// error-pattern:using multiple versions of crate `crateresolve2`

extern crate crateresolve2 = "crateresolve2#0.1";

mod m {
    pub extern crate crateresolve2 = "crateresolve2#0.2";
}

fn main() {
    let x: int = false;
}
