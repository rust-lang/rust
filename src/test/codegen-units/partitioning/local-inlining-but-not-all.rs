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
// compile-flags:-Zprint-trans-items=lazy -Zincremental=tmp/partitioning-tests/local-inlining-but-not-all
// compile-flags:-Zinline-in-all-cgus=no

#![allow(dead_code)]
#![crate_type="lib"]

mod inline {

    //~ TRANS_ITEM fn local_inlining_but_not_all::inline[0]::inlined_function[0] @@ local_inlining_but_not_all-inline[External]
    #[inline]
    pub fn inlined_function()
    {

    }
}

pub mod user1 {
    use super::inline;

    //~ TRANS_ITEM fn local_inlining_but_not_all::user1[0]::foo[0] @@ local_inlining_but_not_all-user1[External]
    pub fn foo() {
        inline::inlined_function();
    }
}

pub mod user2 {
    use super::inline;

    //~ TRANS_ITEM fn local_inlining_but_not_all::user2[0]::bar[0] @@ local_inlining_but_not_all-user2[External]
    pub fn bar() {
        inline::inlined_function();
    }
}

pub mod non_user {

    //~ TRANS_ITEM fn local_inlining_but_not_all::non_user[0]::baz[0] @@ local_inlining_but_not_all-non_user[External]
    pub fn baz() {

    }
}
