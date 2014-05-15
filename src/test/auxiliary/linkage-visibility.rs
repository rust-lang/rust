// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::dynamic_lib::DynamicLibrary;

#[no_mangle]
pub fn foo() { bar(); }

pub fn foo2<T>() {
    fn bar2() {
        bar();
    }
    bar2();
}

#[no_mangle]
fn bar() { }

#[no_mangle]
fn baz() { }

pub fn test() {
    let none: Option<Path> = None; // appease the typechecker
    let lib = DynamicLibrary::open(none).unwrap();
    unsafe {
        assert!(lib.symbol::<int>("foo").is_ok());
        assert!(lib.symbol::<int>("baz").is_err());
        assert!(lib.symbol::<int>("bar").is_err());
    }
}
