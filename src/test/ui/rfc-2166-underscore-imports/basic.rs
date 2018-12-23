// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// aux-build:underscore-imports.rs

#![warn(unused_imports, unused_extern_crates)]

#[macro_use]
extern crate underscore_imports as _;

do_nothing!(); // OK

struct S;

mod m {
    pub trait Tr1 {
        fn tr1_is_in_scope(&self) {}
    }
    pub trait Tr2 {
        fn tr2_is_in_scope(&self) {}
    }

    impl Tr1 for ::S {}
    impl Tr2 for ::S {}
}

mod unused {
    use m::Tr1 as _; //~ WARN unused import
    use S as _; //~ WARN unused import
    extern crate core as _; // OK
}

mod outer {
    mod middle {
        pub use m::Tr1 as _;
        pub use m::Tr2 as _; // OK, no name conflict
        struct Tr1; // OK, no name conflict
        fn check() {
            // Both traits are in scope
            ::S.tr1_is_in_scope();
            ::S.tr2_is_in_scope();
        }

        mod inner {
            // `_` imports are fetched by glob imports
            use super::*;
            fn check() {
                // Both traits are in scope
                ::S.tr1_is_in_scope();
                ::S.tr2_is_in_scope();
            }
        }
    }

    // `_` imports are fetched by glob imports
    use self::middle::*;
    fn check() {
        // Both traits are in scope
        ::S.tr1_is_in_scope();
        ::S.tr2_is_in_scope();
    }
}

fn main() {}
