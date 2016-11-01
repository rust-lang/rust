// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that when instantiating trait default methods, typeck handles
// lifetime parameters defined on the method bound correctly.


pub trait Foo {
    fn bar<'a, I: Iterator<Item=&'a ()>>(&self, it: I) -> usize {
        let mut xs = it.filter(|_| true);
        xs.count()
    }
}

pub struct Baz;

impl Foo for Baz {
    // When instantiating `Foo::bar` for `Baz` here, typeck used to
    // ICE due to the lifetime parameter of `bar`.
}

fn main() {
    let x = Baz;
    let y = vec![(), (), ()];
    assert_eq!(x.bar(y.iter()), 3);
}
