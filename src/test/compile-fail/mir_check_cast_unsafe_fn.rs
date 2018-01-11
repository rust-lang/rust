// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir -Z nll

#![allow(dead_code)]

fn bar<'a>(input: &'a u32, f: fn(&'a u32) -> &'a u32) -> &'static u32 {
    // Here the NLL checker must relate the types in `f` to the types
    // in `g`. These are related via the `UnsafeFnPointer` cast.
    let g: unsafe fn(_) -> _ = f;
    //~^ WARNING not reporting region error due to -Znll
    //~| ERROR free region `'a` does not outlive free region `'static`
    unsafe { g(input) }
}

fn main() {}
