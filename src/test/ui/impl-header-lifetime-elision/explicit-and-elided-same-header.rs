// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![allow(warnings)]

// This works for functions...
fn foo<'a>(x: &str, y: &'a str) {}

// ...so this should work for impls
impl<'a> Foo<&str> for &'a str {}
trait Foo<T> {}

fn main() {
}
