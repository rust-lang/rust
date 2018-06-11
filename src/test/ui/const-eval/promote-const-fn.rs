// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// gate-test-promotable_const_fn

#![feature(const_fn, promotable_const_fn)]

const fn foo() {}

#[promotable_const_fn]
const fn bar() {}

union Foo {
    a: &'static u32,
    b: usize,
}

#[promotable_const_fn]
const fn boo() -> bool {
    unsafe {
        Foo { a: &1 }.b == 42 //~ ERROR promotable constant function contains
    }
}

fn main() {
    let x: &'static () = &foo(); //~ borrowed value does not live long enough
    let x: &'static () = &bar();
}
