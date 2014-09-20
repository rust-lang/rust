// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// forbid-output: in expansion of

#![feature(macro_rules)]

macro_rules! make_method ( ($name:ident) => (
    fn $name(&self) { }
))

struct S;

impl S {
    // We had a bug where these wouldn't clean up macro backtrace frames.
    make_method!(foo1)
    make_method!(foo2)
    make_method!(foo3)
    make_method!(foo4)
    make_method!(foo5)
    make_method!(foo6)
    make_method!(foo7)
    make_method!(foo8)

    // Cause an error. It shouldn't have any macro backtrace frames.
    fn bar(&self) { }
    fn bar(&self) { } //~ ERROR duplicate definition
}

fn main() { }
