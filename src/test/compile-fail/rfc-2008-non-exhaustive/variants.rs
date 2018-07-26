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

/*
 * The initial implementation of #[non_exhaustive] (RFC 2008) does not include support for
 * variants. See issue #44109 and PR 45394.
 */
// ignore-test

fn main() {
    let variant_struct = NonExhaustiveVariants::Struct { field: 640 };
    //~^ ERROR cannot create non-exhaustive variant

    let variant_tuple = NonExhaustiveVariants::Tuple { 0: 640 };
    //~^ ERROR cannot create non-exhaustive variant

    match variant_struct {
        NonExhaustiveVariants::Unit => "",
        NonExhaustiveVariants::Tuple(fe_tpl) => "",
        //~^ ERROR `..` required with variant marked as non-exhaustive
        NonExhaustiveVariants::Struct { field } => ""
        //~^ ERROR `..` required with variant marked as non-exhaustive
    };
}
