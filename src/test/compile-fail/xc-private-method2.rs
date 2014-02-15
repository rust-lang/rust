// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// aux-build:xc_private_method_lib.rs

extern crate xc_private_method_lib;

fn main() {
    let _ = xc_private_method_lib::Struct{ x: 10 }.meth_struct();
    //~^ ERROR method `meth_struct` is private

    let _ = xc_private_method_lib::Variant1(20).meth_enum();
    //~^ ERROR method `meth_enum` is private
}
