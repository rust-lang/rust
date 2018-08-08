// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(stmt_expr_attributes)]

fn main() {

    // Test that attributes on parens get concatenated
    // in the expected order in the hir folder.

    #[deny(non_snake_case)] (
        #![allow(non_snake_case)]
        {
            let X = 0;
            let _ = X;
        }
    );

    #[allow(non_snake_case)] (
        #![deny(non_snake_case)]
        {
            let X = 0; //~ ERROR snake case name
            let _ = X;
        }
    );

}
