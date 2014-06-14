// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:struct_variant_xc_aux.rs
extern crate struct_variant_xc_aux;

use struct_variant_xc_aux::{StructVariant, Variant};

pub fn main() {
    let arg = match (StructVariant { arg: 42 }) {
        Variant(_) => unreachable!(),
        StructVariant { arg } => arg
    };
    assert_eq!(arg, 42);
}
