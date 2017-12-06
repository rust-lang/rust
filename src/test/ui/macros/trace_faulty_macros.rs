// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z trace-macros

#![recursion_limit="4"]

macro_rules! my_faulty_macro {
    () => {
        my_faulty_macro!(bcd); //~ ERROR no rules
    };
}

macro_rules! pat_macro {
    () => {
        pat_macro!(A{a:a, b:0, c:_, ..});
    };
    ($a:pat) => {
        $a
    };
}

macro_rules! my_recursive_macro {
    () => {
        my_recursive_macro!(); //~ ERROR recursion limit
    };
}

macro_rules! my_macro {
    () => {

    };
}

fn main() {
    my_faulty_macro!();
    my_recursive_macro!();
    test!();
    non_exisiting!();
    derive!(Debug);
    let a = pat_macro!();
}

#[my_macro]
fn use_bang_macro_as_attr(){}

#[derive(Debug)]
fn use_derive_macro_as_attr(){}
