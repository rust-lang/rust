// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:private_variant_xc.rs
// xfail-test

extern mod private_variant_xc;

pub fn main() {
    let _ = private_variant_xc::Bar;
    let _ = private_variant_xc::Baz;    //~ ERROR unresolved name
}
