// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private)]

// We're testing linkage visibility; the compiler warns us, but we want to
// do the runtime check that these functions aren't exported.
#![allow(private_no_mangle_fns)]

extern crate rustc_back;

use rustc_back::dynamic_lib::DynamicLibrary;

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

#[allow(dead_code)]
#[no_mangle]
fn baz() { }

pub fn test() {
    let lib = DynamicLibrary::open(None).unwrap();
    unsafe {
        assert!(lib.symbol::<isize>("foo").is_ok());
        assert!(lib.symbol::<isize>("baz").is_err());
        assert!(lib.symbol::<isize>("bar").is_err());
    }
}
