// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:enums.rs
extern crate enums;

// ignore-pretty issue #37199

use enums::NonExhaustiveEnum;

fn main() {
    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        NonExhaustiveEnum::Unit => 1,
        NonExhaustiveEnum::Tuple(_) => 2,
        // This particular arm tests that a enum marked as non-exhaustive
        // will not error if its variants are matched exhaustively.
        NonExhaustiveEnum::Struct { field } => field,
        _ => 0 // no error with wildcard
    };

    match enum_unit {
        _ => "no error with only wildcard"
    };
}
