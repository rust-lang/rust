// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

struct Misc<T:?Sized>(T);

fn check<T: Iterator, U: ?Sized>() {
    // suggest a where-clause, if needed
    mem::size_of::<U>();
    //~^ ERROR the size for values of type

    mem::size_of::<Misc<U>>();
    //~^ ERROR the size for values of type

    // ... even if T occurs as a type parameter

    <u64 as From<T>>::from;
    //~^ ERROR `u64: std::convert::From<T>` is not satisfied

    <u64 as From<<T as Iterator>::Item>>::from;
    //~^ ERROR `u64: std::convert::From<<T as std::iter::Iterator>::Item>` is not satisfied

    // ... but not if there are inference variables

    <Misc<_> as From<T>>::from;
    //~^ ERROR `Misc<_>: std::convert::From<T>` is not satisfied

    // ... and also not if the error is not related to the type

    mem::size_of::<[T]>();
    //~^ ERROR the size for values of type

    mem::size_of::<[&U]>();
    //~^ ERROR the size for values of type
}

fn main() {
}
