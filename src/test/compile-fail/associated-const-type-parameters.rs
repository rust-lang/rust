// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

pub trait Foo {
    const MIN: i32;

    fn get_min() -> i32 {
        Self::MIN //~ ERROR E0329
    }
}

fn get_min<T: Foo>() -> i32 {
    T::MIN; //~ ERROR E0329
    <T as Foo>::MIN //~ ERROR E0329
}

fn main() {}
