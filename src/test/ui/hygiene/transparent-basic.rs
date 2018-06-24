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
// aux-build:transparent-basic.rs

#![feature(decl_macro, rustc_attrs)]

extern crate transparent_basic;

#[rustc_transparent_macro]
macro binding() {
    let x = 10;
}

#[rustc_transparent_macro]
macro label() {
    break 'label
}

macro_rules! legacy {
    () => {
        binding!();
        let y = x;
    }
}

fn legacy_interaction1() {
    legacy!();
}

struct S;

fn check_dollar_crate() {
    // `$crate::S` inside the macro resolves to `S` from this crate.
    transparent_basic::dollar_crate!();
}

fn main() {
    binding!();
    let y = x;

    'label: loop {
        label!();
    }
}
