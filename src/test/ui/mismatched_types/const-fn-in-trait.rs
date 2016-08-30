// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustc-env:RUST_NEW_ERROR_FORMAT

#![feature(const_fn)]

trait Foo {
    fn f() -> u32;
    const fn g();
}

impl Foo for u32 {
    const fn f() -> u32 { 22 }
    fn g() {}
}

fn main() { }
