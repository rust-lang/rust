// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub mod b1 {
        pub enum C2 {}
    }

    pub enum B2 {}
}

use a::{b1::{C1, C2}, B2};
//~^ ERROR unresolved import `a::b1::C1`

fn main() {
    let _: C2;
    let _: B2;
}
