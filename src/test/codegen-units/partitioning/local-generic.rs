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
// compile-flags:-Zprint-trans-items=eager -Zincremental=tmp/partitioning-tests/local-generic

#![allow(dead_code)]
#![crate_type="lib"]

//~ TRANS_ITEM fn local_generic::generic[0]<u32> @@ local_generic.volatile[External]
//~ TRANS_ITEM fn local_generic::generic[0]<u64> @@ local_generic.volatile[External]
//~ TRANS_ITEM fn local_generic::generic[0]<char> @@ local_generic.volatile[External]
//~ TRANS_ITEM fn local_generic::generic[0]<&str> @@ local_generic.volatile[External]
pub fn generic<T>(x: T) -> T { x }

//~ TRANS_ITEM fn local_generic::user[0] @@ local_generic[External]
fn user() {
    let _ = generic(0u32);
}

mod mod1 {
    pub use super::generic;

    //~ TRANS_ITEM fn local_generic::mod1[0]::user[0] @@ local_generic-mod1[External]
    fn user() {
        let _ = generic(0u64);
    }

    mod mod1 {
        use super::generic;

        //~ TRANS_ITEM fn local_generic::mod1[0]::mod1[0]::user[0] @@ local_generic-mod1-mod1[External]
        fn user() {
            let _ = generic('c');
        }
    }
}

mod mod2 {
    use super::generic;

    //~ TRANS_ITEM fn local_generic::mod2[0]::user[0] @@ local_generic-mod2[External]
    fn user() {
        let _ = generic("abc");
    }
}
