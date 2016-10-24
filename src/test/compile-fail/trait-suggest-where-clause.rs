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
    //~^ ERROR `U: std::marker::Sized` is not satisfied
    //~| NOTE the trait `std::marker::Sized` is not implemented for `U`
    //~| HELP consider adding a `where U: std::marker::Sized` bound
    //~| NOTE required by `std::mem::size_of`

    mem::size_of::<Misc<U>>();
    //~^ ERROR `U: std::marker::Sized` is not satisfied
    //~| NOTE the trait `std::marker::Sized` is not implemented for `U`
    //~| HELP consider adding a `where U: std::marker::Sized` bound
    //~| NOTE required because it appears within the type `Misc<U>`
    //~| NOTE required by `std::mem::size_of`

    // ... even if T occurs as a type parameter

    <u64 as From<T>>::from;
    //~^ ERROR `u64: std::convert::From<T>` is not satisfied
    //~| NOTE the trait `std::convert::From<T>` is not implemented for `u64`
    //~| HELP consider adding a `where u64: std::convert::From<T>` bound
    //~| NOTE required by `std::convert::From::from`

    <u64 as From<<T as Iterator>::Item>>::from;
    //~^ ERROR `u64: std::convert::From<<T as std::iter::Iterator>::Item>` is not satisfied
    //~| NOTE the trait `std::convert::From<<T as std::iter::Iterator>::Item>` is not implemented
    //~| HELP consider adding a `where u64:
    //~| NOTE required by `std::convert::From::from`

    // ... but not if there are inference variables

    <Misc<_> as From<T>>::from;
    //~^ ERROR `Misc<_>: std::convert::From<T>` is not satisfied
    //~| NOTE the trait `std::convert::From<T>` is not implemented for `Misc<_>`
    //~| NOTE required by `std::convert::From::from`

    // ... and also not if the error is not related to the type

    mem::size_of::<[T]>();
    //~^ ERROR `[T]: std::marker::Sized` is not satisfied
    //~| NOTE the trait `std::marker::Sized` is not implemented for `[T]`
    //~| NOTE `[T]` does not have a constant size
    //~| NOTE required by `std::mem::size_of`

    mem::size_of::<[&U]>();
    //~^ ERROR `[&U]: std::marker::Sized` is not satisfied
    //~| NOTE the trait `std::marker::Sized` is not implemented for `[&U]`
    //~| NOTE `[&U]` does not have a constant size
    //~| NOTE required by `std::mem::size_of`
}

fn main() {
}
