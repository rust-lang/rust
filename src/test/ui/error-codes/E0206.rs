// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type Foo = [u8; 256];

impl Copy for Foo { }
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types

#[derive(Copy, Clone)]
struct Bar;

impl Copy for &'static mut Bar { }
//~^ ERROR the trait `Copy` may not be implemented for this type

fn main() {
}
