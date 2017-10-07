// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty pretty-printing is unhygienic

#![feature(decl_macro)]
#![allow(unused)]

mod ok {
    macro mac_trait_item($method: ident) {
        fn $method();
    }

    trait Tr {
        mac_trait_item!(method);
    }

    macro mac_trait_impl() {
        impl Tr for u8 { // OK
            fn method() {} // OK
        }
    }

    mac_trait_impl!();
}

mod error {
    macro mac_trait_item() {
        fn method();
    }

    trait Tr {
        mac_trait_item!();
    }

    macro mac_trait_impl() {
        impl Tr for u8 { //~ ERROR not all trait items implemented, missing: `method`
            fn method() {} //~ ERROR method `method` is not a member of trait `Tr`
        }
    }

    mac_trait_impl!();
}

fn main() {}
