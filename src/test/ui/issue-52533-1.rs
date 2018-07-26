// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(warnings)]

struct Foo<'a, 'b, T: 'a + 'b> { x: &'a T, y: &'b T }

fn gimme(_: impl for<'a, 'b, 'c> FnOnce(&'a Foo<'a, 'b, u32>,
                                        &'a Foo<'a, 'c, u32>) -> &'a Foo<'a, 'b, u32>) { }

fn main() {
    gimme(|x, y| y)
    //~^ ERROR mismatched types [E0308]
}
