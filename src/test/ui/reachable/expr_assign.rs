// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]
#![feature(never_type)]

fn foo() {
    // No error here.
    let x;
    x = return; //~ ERROR unreachable
}

fn bar() {
    use std::ptr;
    let p: *mut ! = ptr::null_mut::<!>();
    unsafe {
        // Here we consider the `return` unreachable because
        // "evaluating" the `*p` has type `!`. This is somewhat
        // dubious, I suppose.
        *p = return; //~ ERROR unreachable
    }
}

fn baz() {
    let mut i = 0;
    *{return; &mut i} = 22; //~ ERROR unreachable
}

fn main() { }
