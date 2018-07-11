// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is an equivalent of issue #50504, but for declarative macros.

#![feature(decl_macro, rustc_attrs)]

macro genmod($FromOutside: ident, $Outer: ident) {
    type A = $FromOutside;
    struct $Outer;
    mod inner {
        type A = $FromOutside; // `FromOutside` shouldn't be available from here
        type Inner = $Outer; // `Outer` shouldn't be available from here
    }
}

#[rustc_transparent_macro]
macro genmod_transparent() {
    type A = FromOutside;
    struct Outer;
    mod inner {
        type A = FromOutside; //~ ERROR cannot find type `FromOutside` in this scope
        type Inner = Outer; //~ ERROR cannot find type `Outer` in this scope
    }
}

macro_rules! genmod_legacy { () => {
    type A = FromOutside;
    struct Outer;
    mod inner {
        type A = FromOutside; //~ ERROR cannot find type `FromOutside` in this scope
        type Inner = Outer; //~ ERROR cannot find type `Outer` in this scope
    }
}}

fn check() {
    struct FromOutside;
    genmod!(FromOutside, Outer); //~ ERROR cannot find type `FromOutside` in this scope
                                 //~| ERROR cannot find type `Outer` in this scope
}

fn check_transparent() {
    struct FromOutside;
    genmod_transparent!();
}

fn check_legacy() {
    struct FromOutside;
    genmod_legacy!();
}
