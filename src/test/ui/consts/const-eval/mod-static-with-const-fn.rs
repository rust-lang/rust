// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// New test for #53818: modifying static memory at compile-time is not allowed.
// The test should never compile successfully

#![feature(const_raw_ptr_deref)]
#![feature(const_let)]

use std::cell::UnsafeCell;

struct Foo(UnsafeCell<u32>);

unsafe impl Send for Foo {}
unsafe impl Sync for Foo {}

static FOO: Foo = Foo(UnsafeCell::new(42));

fn foo() {}

static BAR: () = unsafe {
    *FOO.0.get() = 5;
    //~^ contains unimplemented expression

    foo();
    //~^ ERROR calls in statics are limited to constant functions, tuple structs and tuple variants
};

fn main() {
    println!("{}", unsafe { *FOO.0.get() });
}
