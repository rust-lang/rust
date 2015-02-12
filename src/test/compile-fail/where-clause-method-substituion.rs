// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo<T> {
    fn dummy(&self, t: T) { }
}

trait Bar<A> {
    fn method<B>(&self) where A: Foo<B>;
}

struct S;
struct X;

// Remove this impl causing the below resolution to fail // impl Foo<S> for X {}

impl Bar<X> for isize {
    fn method<U>(&self) where X: Foo<U> {
    }
}

fn main() {
    1.method::<X>();
    //~^ ERROR the trait `Foo<X>` is not implemented for the type `X`
}
