// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #37154: the problem here was that the cache
// results in a false error because it was caching skolemized results
// even after those skolemized regions had been popped.

trait Foo {
    fn method(&self) {}
}

struct Wrapper<T>(T);

impl<T> Foo for Wrapper<T> where for<'a> &'a T: IntoIterator<Item=&'a ()> {}

fn f(x: Wrapper<Vec<()>>) {
    x.method(); // This works.
    x.method(); // error: no method named `method`
}

fn main() { }
