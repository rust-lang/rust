// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:attribute-with-error.rs
// ignore-stage1

#![feature(proc_macro)]

extern crate attribute_with_error;

use attribute_with_error::foo;

#[foo]
fn test1() {
    let a: i32 = "foo";
    //~^ ERROR: mismatched types
    let b: i32 = "f'oo";
    //~^ ERROR: mismatched types
}

fn test2() {
    #![foo]

    // FIXME: should have a type error here and assert it works but it doesn't
}

trait A {
    // FIXME: should have a #[foo] attribute here and assert that it works
    fn foo(&self) {
        let a: i32 = "foo";
        //~^ ERROR: mismatched types
    }
}

struct B;

impl A for B {
    #[foo]
    fn foo(&self) {
        let a: i32 = "foo";
        //~^ ERROR: mismatched types
    }
}

#[foo]
fn main() {
}
