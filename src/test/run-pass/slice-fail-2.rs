// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that is a slicing expr[..] fails, the correct cleanups happen.

#![feature(slicing_syntax)]

use std::task;

struct Foo;

static mut DTOR_COUNT: int = 0;

impl Drop for Foo {
    fn drop(&mut self) { unsafe { DTOR_COUNT += 1; } }
}

fn bar() -> uint {
    fail!();
}

fn foo() {
    let x: &[_] = &[Foo, Foo];
    x[3..bar()];
}

fn main() {
    let _ = task::try(proc() foo());
    unsafe { assert!(DTOR_COUNT == 2); }
}
