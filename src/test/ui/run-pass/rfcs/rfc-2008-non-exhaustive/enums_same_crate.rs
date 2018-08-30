// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(non_exhaustive)]

#[non_exhaustive]
pub enum NonExhaustiveEnum {
    Unit,
    Tuple(u32),
    Struct { field: u32 }
}

fn main() {
    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        NonExhaustiveEnum::Unit => "first",
        NonExhaustiveEnum::Tuple(_) => "second",
        NonExhaustiveEnum::Struct { .. } => "third",
    };
}
