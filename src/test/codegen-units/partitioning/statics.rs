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
// compile-flags:-Zprint-trans-items=lazy

#![crate_type="lib"]

//~ TRANS_ITEM static statics::FOO[0] @@ statics[External]
static FOO: u32 = 0;

//~ TRANS_ITEM static statics::BAR[0] @@ statics[External]
static BAR: u32 = 0;

//~ TRANS_ITEM fn statics::function[0] @@ statics[WeakODR]
fn function() {
    //~ TRANS_ITEM static statics::function[0]::FOO[0] @@ statics[External]
    static FOO: u32 = 0;

    //~ TRANS_ITEM static statics::function[0]::BAR[0] @@ statics[External]
    static BAR: u32 = 0;
}

mod mod1 {
    //~ TRANS_ITEM static statics::mod1[0]::FOO[0] @@ statics-mod1[External]
    static FOO: u32 = 0;

    //~ TRANS_ITEM static statics::mod1[0]::BAR[0] @@ statics-mod1[External]
    static BAR: u32 = 0;

    //~ TRANS_ITEM fn statics::mod1[0]::function[0] @@ statics-mod1[WeakODR]
    fn function() {
        //~ TRANS_ITEM static statics::mod1[0]::function[0]::FOO[0] @@ statics-mod1[External]
        static FOO: u32 = 0;

        //~ TRANS_ITEM static statics::mod1[0]::function[0]::BAR[0] @@ statics-mod1[External]
        static BAR: u32 = 0;
    }
}

//~ TRANS_ITEM drop-glue i8
