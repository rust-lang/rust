// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:import_crate_var.rs

// compile-pass
// skip-codegen
#[macro_use] extern crate import_crate_var;


fn main() {
    m!();
    //~^ WARN `$crate` may not be imported
    //~| NOTE `use $crate;` was erroneously allowed and will become a hard error
}
