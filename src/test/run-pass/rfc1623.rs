// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

// very simple test for a 'static static with default lifetime
static SOME_STATIC_STR : &str = "&'static str";
const SOME_CONST_STR : &str = "&'static str";

// this should be the same as without default:
static SOME_EXPLICIT_STATIC_STR : &'static str = "&'static str";
const SOME_EXPLICIT_CONST_STR : &'static str = "&'static str";

// a function that elides to an unbound lifetime for both in- and output
fn id_u8_slice(arg: &[u8]) -> &[u8] { arg }

// one with a function, argument elided
static SOME_STATIC_SIMPLE_FN : &fn(&[u8]) -> &[u8] =
        &(id_u8_slice as fn(&[u8]) -> &[u8]);
const SOME_CONST_SIMPLE_FN : &fn(&[u8]) -> &[u8] =
        &(id_u8_slice as fn(&[u8]) -> &[u8]);

// this should be the same as without elision
static SOME_STATIC_NON_ELIDED_fN : &for<'a> fn(&'a [u8]) -> &'a [u8] =
        &(id_u8_slice as for<'a> fn(&'a [u8]) -> &'a [u8]);
const SOME_CONST_NON_ELIDED_fN : &for<'a> fn(&'a [u8]) -> &'a [u8] =
        &(id_u8_slice as for<'a> fn(&'a [u8]) -> &'a [u8]);

// another function that elides, each to a different unbound lifetime
fn multi_args(a: &u8, b: &u8, c: &u8) { }

static SOME_STATIC_MULTI_FN : &fn(&u8, &u8, &u8) =
        &(multi_args as fn(&u8, &u8, &u8));
const SOME_CONST_MULTI_FN : &fn(&u8, &u8, &u8) =
        &(multi_args as fn(&u8, &u8, &u8));


fn main() {
    // make sure that the lifetime is actually elided (and not defaulted)
    let x = &[1u8, 2, 3];
    SOME_STATIC_SIMPLE_FN(x);
    SOME_CONST_SIMPLE_FN(x);

    // make sure this works with different lifetimes
    let a = &1;
    {
        let b = &2;
        let c = &3;
        SOME_CONST_MULTI_FN(a, b, c);
    }
}
