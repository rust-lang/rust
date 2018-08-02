// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro-in-other-crate.rs

#![feature(decl_macro, extern_prelude)]

macro_rules! add_macro_expanded_things_to_macro_prelude {() => {
    #[macro_use]
    extern crate macro_in_other_crate;
}}

add_macro_expanded_things_to_macro_prelude!();

mod m1 {
    fn check() {
        inline!(); //~ ERROR `inline` is ambiguous
    }
}

mod m2 {
    pub mod std {
        pub macro panic() {}
    }
}

mod m3 {
    use m2::*; // glob-import user-defined `std`
    fn check() {
        std::panic!(); //~ ERROR `std` is ambiguous
    }
}

fn main() {}
