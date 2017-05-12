// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that specialization must be ungated to use the `default` keyword

// gate-test-specialization

trait Foo {
    fn foo(&self);
}

impl<T> Foo for T {
    default fn foo(&self) {} //~ ERROR specialization is unstable
}

fn main() {}
