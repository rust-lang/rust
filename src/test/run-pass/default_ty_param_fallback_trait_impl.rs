// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(default_type_parameter_fallback)]

// Another example from the RFC
trait Foo { }
trait Bar { }

impl<T:Bar=usize> Foo for Vec<T> {}
impl Bar for usize {}

fn takes_foo<F:Foo>(f: F) {}

fn main() {
    let x = Vec::new(); // x: Vec<$0>
    takes_foo(x); // adds oblig Vec<$0> : Foo
}
