// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]

extern crate core;
pub use core as reexported_core; //~ ERROR `core` is private, and cannot be re-exported
                                 //~^ WARN this was previously accepted

mod foo1 {
    extern crate core;
}

mod foo2 {
    use foo1::core; //~ ERROR `core` is private, and cannot be re-exported
                    //~^ WARN this was previously accepted
    pub mod bar {
        extern crate core;
    }
}

mod baz {
    pub use foo2::bar::core; //~ ERROR `core` is private, and cannot be re-exported
                             //~^ WARN this was previously accepted
}

fn main() {}
