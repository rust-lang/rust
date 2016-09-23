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
// compile-flags:-Zprint-trans-items=lazy -Zincremental=tmp/partitioning-tests/local-inlining

#![allow(dead_code)]
#![crate_type="lib"]

mod inline {

    // Important: This function should show up in all codegen units where it is inlined
    //~ TRANS_ITEM fn local_inlining::inline[0]::inlined_function[0] @@ local_inlining-user1[Internal] local_inlining-user2[Internal]
    #[inline(always)]
    pub fn inlined_function()
    {

    }
}

mod user1 {
    use super::inline;

    //~ TRANS_ITEM fn local_inlining::user1[0]::foo[0] @@ local_inlining-user1[External]
    fn foo() {
        inline::inlined_function();
    }
}

mod user2 {
    use super::inline;

    //~ TRANS_ITEM fn local_inlining::user2[0]::bar[0] @@ local_inlining-user2[External]
    fn bar() {
        inline::inlined_function();
    }
}

mod non_user {

    //~ TRANS_ITEM fn local_inlining::non_user[0]::baz[0] @@ local_inlining-non_user[External]
    fn baz() {

    }
}
