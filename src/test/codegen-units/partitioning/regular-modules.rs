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
// compile-flags:-Zprint-mono-items=eager -Zincremental=tmp/partitioning-tests/regular-modules

#![allow(dead_code)]
#![crate_type="lib"]

//~ MONO_ITEM fn regular_modules::foo[0] @@ regular_modules[Internal]
fn foo() {}

//~ MONO_ITEM fn regular_modules::bar[0] @@ regular_modules[Internal]
fn bar() {}

//~ MONO_ITEM static regular_modules::BAZ[0] @@ regular_modules[Internal]
static BAZ: u64 = 0;

mod mod1 {

    //~ MONO_ITEM fn regular_modules::mod1[0]::foo[0] @@ regular_modules-mod1[Internal]
    fn foo() {}
    //~ MONO_ITEM fn regular_modules::mod1[0]::bar[0] @@ regular_modules-mod1[Internal]
    fn bar() {}
    //~ MONO_ITEM static regular_modules::mod1[0]::BAZ[0] @@ regular_modules-mod1[Internal]
    static BAZ: u64 = 0;

    mod mod1 {
        //~ MONO_ITEM fn regular_modules::mod1[0]::mod1[0]::foo[0] @@ regular_modules-mod1-mod1[Internal]
        fn foo() {}
        //~ MONO_ITEM fn regular_modules::mod1[0]::mod1[0]::bar[0] @@ regular_modules-mod1-mod1[Internal]
        fn bar() {}
        //~ MONO_ITEM static regular_modules::mod1[0]::mod1[0]::BAZ[0] @@ regular_modules-mod1-mod1[Internal]
        static BAZ: u64 = 0;
    }

    mod mod2 {
        //~ MONO_ITEM fn regular_modules::mod1[0]::mod2[0]::foo[0] @@ regular_modules-mod1-mod2[Internal]
        fn foo() {}
        //~ MONO_ITEM fn regular_modules::mod1[0]::mod2[0]::bar[0] @@ regular_modules-mod1-mod2[Internal]
        fn bar() {}
        //~ MONO_ITEM static regular_modules::mod1[0]::mod2[0]::BAZ[0] @@ regular_modules-mod1-mod2[Internal]
        static BAZ: u64 = 0;
    }
}

mod mod2 {

    //~ MONO_ITEM fn regular_modules::mod2[0]::foo[0] @@ regular_modules-mod2[Internal]
    fn foo() {}
    //~ MONO_ITEM fn regular_modules::mod2[0]::bar[0] @@ regular_modules-mod2[Internal]
    fn bar() {}
    //~ MONO_ITEM static regular_modules::mod2[0]::BAZ[0] @@ regular_modules-mod2[Internal]
    static BAZ: u64 = 0;

    mod mod1 {
        //~ MONO_ITEM fn regular_modules::mod2[0]::mod1[0]::foo[0] @@ regular_modules-mod2-mod1[Internal]
        fn foo() {}
        //~ MONO_ITEM fn regular_modules::mod2[0]::mod1[0]::bar[0] @@ regular_modules-mod2-mod1[Internal]
        fn bar() {}
        //~ MONO_ITEM static regular_modules::mod2[0]::mod1[0]::BAZ[0] @@ regular_modules-mod2-mod1[Internal]
        static BAZ: u64 = 0;
    }

    mod mod2 {
        //~ MONO_ITEM fn regular_modules::mod2[0]::mod2[0]::foo[0] @@ regular_modules-mod2-mod2[Internal]
        fn foo() {}
        //~ MONO_ITEM fn regular_modules::mod2[0]::mod2[0]::bar[0] @@ regular_modules-mod2-mod2[Internal]
        fn bar() {}
        //~ MONO_ITEM static regular_modules::mod2[0]::mod2[0]::BAZ[0] @@ regular_modules-mod2-mod2[Internal]
        static BAZ: u64 = 0;
    }
}
