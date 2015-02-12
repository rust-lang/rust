// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![allow(warnings)]

use std::marker::PhantomFn;

trait A<T> : PhantomFn<(Self,T)> {}
struct B<'a, T>(&'a (A<T>+'a));

trait X : PhantomFn<Self> {}
impl<'a, T> X for B<'a, T> {}

fn h<'a, T, U>(v: Box<A<U>+'static>) -> Box<X+'static> {
    box B(&*v) as Box<X> //~ ERROR `*v` does not live long enough
}

fn main() {}

