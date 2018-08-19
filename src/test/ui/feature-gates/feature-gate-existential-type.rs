// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that existential types must be ungated to use the `existential` keyword



existential type Foo: std::fmt::Debug; //~ ERROR existential types are unstable

trait Bar {
    type Baa: std::fmt::Debug;
}

impl Bar for () {
    existential type Baa: std::fmt::Debug; //~ ERROR existential types are unstable
}

fn main() {}
