// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

macro_rules! foo { () => {
    let x = 1;
    macro_rules! bar { () => {x} }
    let _ = bar!();
}}

macro_rules! m { // test issue #31856
    ($n:ident) => (
        let a = 1;
        let $n = a;
    )
}

macro_rules! baz {
    ($i:ident) => {
        let mut $i = 2;
        $i = $i + 1;
    }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    foo! {};
    bar! {};

    let mut a = true;
    baz!(a);
}
