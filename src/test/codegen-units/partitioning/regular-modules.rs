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
// compile-flags:-Zprint-trans-items=eager -Zincremental=tmp/partitioning-tests/regular-modules

#![allow(dead_code)]
#![crate_type="lib"]

//~ TRANS_ITEM fn regular_modules::foo[0] @@ regular_modules[External]
fn foo() {}

//~ TRANS_ITEM fn regular_modules::bar[0] @@ regular_modules[External]
fn bar() {}

//~ TRANS_ITEM static regular_modules::BAZ[0] @@ regular_modules[External]
static BAZ: u64 = 0;

mod mod1 {

    //~ TRANS_ITEM fn regular_modules::mod1[0]::foo[0] @@ regular_modules-mod1[External]
    fn foo() {}
    //~ TRANS_ITEM fn regular_modules::mod1[0]::bar[0] @@ regular_modules-mod1[External]
    fn bar() {}
    //~ TRANS_ITEM static regular_modules::mod1[0]::BAZ[0] @@ regular_modules-mod1[External]
    static BAZ: u64 = 0;

    mod mod1 {
        //~ TRANS_ITEM fn regular_modules::mod1[0]::mod1[0]::foo[0] @@ regular_modules-mod1-mod1[External]
        fn foo() {}
        //~ TRANS_ITEM fn regular_modules::mod1[0]::mod1[0]::bar[0] @@ regular_modules-mod1-mod1[External]
        fn bar() {}
        //~ TRANS_ITEM static regular_modules::mod1[0]::mod1[0]::BAZ[0] @@ regular_modules-mod1-mod1[External]
        static BAZ: u64 = 0;
    }

    mod mod2 {
        //~ TRANS_ITEM fn regular_modules::mod1[0]::mod2[0]::foo[0] @@ regular_modules-mod1-mod2[External]
        fn foo() {}
        //~ TRANS_ITEM fn regular_modules::mod1[0]::mod2[0]::bar[0] @@ regular_modules-mod1-mod2[External]
        fn bar() {}
        //~ TRANS_ITEM static regular_modules::mod1[0]::mod2[0]::BAZ[0] @@ regular_modules-mod1-mod2[External]
        static BAZ: u64 = 0;
    }
}

mod mod2 {

    //~ TRANS_ITEM fn regular_modules::mod2[0]::foo[0] @@ regular_modules-mod2[External]
    fn foo() {}
    //~ TRANS_ITEM fn regular_modules::mod2[0]::bar[0] @@ regular_modules-mod2[External]
    fn bar() {}
    //~ TRANS_ITEM static regular_modules::mod2[0]::BAZ[0] @@ regular_modules-mod2[External]
    static BAZ: u64 = 0;

    mod mod1 {
        //~ TRANS_ITEM fn regular_modules::mod2[0]::mod1[0]::foo[0] @@ regular_modules-mod2-mod1[External]
        fn foo() {}
        //~ TRANS_ITEM fn regular_modules::mod2[0]::mod1[0]::bar[0] @@ regular_modules-mod2-mod1[External]
        fn bar() {}
        //~ TRANS_ITEM static regular_modules::mod2[0]::mod1[0]::BAZ[0] @@ regular_modules-mod2-mod1[External]
        static BAZ: u64 = 0;
    }

    mod mod2 {
        //~ TRANS_ITEM fn regular_modules::mod2[0]::mod2[0]::foo[0] @@ regular_modules-mod2-mod2[External]
        fn foo() {}
        //~ TRANS_ITEM fn regular_modules::mod2[0]::mod2[0]::bar[0] @@ regular_modules-mod2-mod2[External]
        fn bar() {}
        //~ TRANS_ITEM static regular_modules::mod2[0]::mod2[0]::BAZ[0] @@ regular_modules-mod2-mod2[External]
        static BAZ: u64 = 0;
    }
}

//~ TRANS_ITEM drop-glue i8
