// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait A<T> {}
struct B<'a, T>(&'a A<T>);

trait X {}
impl<'a, T> X for B<'a, T> {}

fn f<'a, T, U>(v: Box<A<T>>) -> Box<X:> {
    box B(v) as Box<X:> //~ ERROR value may contain references; add `'static` bound to `T`
}

fn g<'a, T, U>(v: Box<A<U>>) -> Box<X:> {
    box B(v) as Box<X:> //~ ERROR value may contain references; add `'static` bound to `U`
}

fn h<'a, T: 'static>(v: Box<A<T>>) -> Box<X:> {
    box B(v) as Box<X:> // ok
}

fn main() {}

