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

// An example from the RFC
trait Foo { fn takes_foo(&self); }
trait Bar { }

impl<T:Bar=usize> Foo for Vec<T> {
    fn takes_foo(&self) {}
}

impl Bar for usize {}

fn main() {
    let x = Vec::new(); // x: Vec<$0>
    x.takes_foo(); // adds oblig Vec<$0> : Foo
}
