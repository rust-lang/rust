// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// aux-build:macro-in-other-crate.rs

#![feature(decl_macro)]

macro_rules! my_include {() => {
    // Outer
    macro m() {}
    #[macro_use(from_prelude)] extern crate macro_in_other_crate;

    fn inner() {
        // Inner
        macro m() {}
        macro_rules! from_prelude { () => {} }

        // OK, both `m` and `from_prelude` are macro-expanded,
        // but no more macro-expanded than their counterpart from outer scope.
        m!();
        from_prelude!();
    }
}}

my_include!();

fn main() {}
