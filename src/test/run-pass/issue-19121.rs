// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a partially specified trait object with unspecified associated
// type does not ICE.

trait Foo {
    type A;

    fn dummy(&self) { }
}

fn bar(x: &Foo) {}
// FIXME(#19482) -- `Foo` should specify `A`, but this is not
// currently enforced except at object creation

pub fn main() {}
