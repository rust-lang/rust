// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-trans-items=eager -Zincremental=tmp/partitioning-tests/extern-generic

#![allow(dead_code)]
#![crate_type="lib"]

// aux-build:cgu_generic_function.rs
extern crate cgu_generic_function;

//~ TRANS_ITEM fn extern_generic::user[0] @@ extern_generic[External]
fn user() {
    let _ = cgu_generic_function::foo("abc");
}

mod mod1 {
    use cgu_generic_function;

    //~ TRANS_ITEM fn extern_generic::mod1[0]::user[0] @@ extern_generic-mod1[External]
    fn user() {
        let _ = cgu_generic_function::foo("abc");
    }

    mod mod1 {
        use cgu_generic_function;

        //~ TRANS_ITEM fn extern_generic::mod1[0]::mod1[0]::user[0] @@ extern_generic-mod1-mod1[External]
        fn user() {
            let _ = cgu_generic_function::foo("abc");
        }
    }
}

mod mod2 {
    use cgu_generic_function;

    //~ TRANS_ITEM fn extern_generic::mod2[0]::user[0] @@ extern_generic-mod2[External]
    fn user() {
        let _ = cgu_generic_function::foo("abc");
    }
}

mod mod3 {
    //~ TRANS_ITEM fn extern_generic::mod3[0]::non_user[0] @@ extern_generic-mod3[External]
    fn non_user() {}
}

// Make sure the two generic functions from the extern crate get instantiated
// once for the current crate
//~ TRANS_ITEM fn cgu_generic_function::foo[0]<&str> @@ cgu_generic_function.volatile[External]
//~ TRANS_ITEM fn cgu_generic_function::bar[0]<&str> @@ cgu_generic_function.volatile[External]

//~ TRANS_ITEM drop-glue i8
