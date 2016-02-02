// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    extern crate core;
    pub use self::core as reexported_core; // Check that private extern crates can be reexported
}

// Check that private crates cannot be used from outside their modules
use foo::core; //~ ERROR module `core` is inaccessible
use foo::core::cell; //~ ERROR

fn main() {
    use foo::*;
    mod core {} // Check that private crates are not glob imported
}
