// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:variants.rs
extern crate variants;

use variants::NonExhaustiveVariants;

// We only test matching here as we cannot create non-exhaustive
// variants from another crate. ie. they'll never pass in run-pass tests.

fn match_variants(non_exhaustive_enum: NonExhaustiveVariants) {
    match non_exhaustive_enum {
        NonExhaustiveVariants::Unit => "",
        NonExhaustiveVariants::Struct { field, .. } => "",
        NonExhaustiveVariants::Tuple(fe_tpl, ..) => ""
    };
}

fn main() {}
