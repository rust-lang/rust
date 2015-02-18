// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the compiler considers the 'static bound declared in the
// trait. Issue #20890.

trait Foo {
    type Value: 'static;
    fn dummy(&self) { }
}

fn require_static<T: 'static>() {}

fn takes_foo<F: Foo>() {
    require_static::<F::Value>()
}

fn main() { }
