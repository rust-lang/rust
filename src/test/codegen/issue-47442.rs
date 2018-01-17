// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that we don't emit unneeded `resume` cleanup blocks for every
// destructor.

// CHECK-NOT: Unwind

#![feature(test)]
#![crate_type="rlib"]

extern crate test;

struct Foo {}

impl Drop for Foo {
    fn drop(&mut self) {
        test::black_box(());
    }
}

#[no_mangle]
pub fn foo() {
    let _foo = Foo {};
}
