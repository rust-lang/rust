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
#![allow(dead_code)]

macro_rules! foo {
    ($x:tt) => (type Alias = $x<i32>;)
}

foo!(Box);

macro_rules! bar {
    ($x:tt) => {
        macro_rules! baz {
            ($x:tt, $y:tt) => { ($x, $y) }
        }
    }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    bar!($y);
    let _: (i8, i16) = baz!(0i8, 0i16);
}
