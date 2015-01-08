// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![deny(raw_pointer_derive)]

#[derive(Clone)]
struct Foo {
    x: *const isize //~ ERROR use of `#[derive]` with a raw pointer
}

#[derive(Clone)]
struct Bar(*mut isize); //~ ERROR use of `#[derive]` with a raw pointer

#[derive(Clone)]
enum Baz {
    A(*const isize), //~ ERROR use of `#[derive]` with a raw pointer
    B { x: *mut isize } //~ ERROR use of `#[derive]` with a raw pointer
}

#[derive(Clone)]
struct Buzz {
    x: (*const isize, //~ ERROR use of `#[derive]` with a raw pointer
        *const usize) //~ ERROR use of `#[derive]` with a raw pointer
}

fn main() {}
