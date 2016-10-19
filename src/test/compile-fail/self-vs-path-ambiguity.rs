// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `self::foo` is parsed as a general pattern and not a self argument.

struct S;

impl S {
    fn f(self::S: S) {}
    fn g(&self::S: &S) {}
    fn h(&mut self::S: &mut S) {}
    fn i(&'a self::S: &S) {} //~ ERROR unexpected lifetime `'a` in pattern
                             //~^ ERROR expected one of `)` or `mut`, found `'a`
}

fn main() {}
