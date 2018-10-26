// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsize, dispatch_from_dyn)]

use std::{
    ops::DispatchFromDyn,
    marker::{Unsize, PhantomData},
};

struct WrapperWithExtraField<T>(T, i32);

impl<T, U> DispatchFromDyn<WrapperWithExtraField<U>> for WrapperWithExtraField<T>
where
    T: DispatchFromDyn<U>,
{} //~^^^ ERROR [E0378]


struct MultiplePointers<T: ?Sized>{
    ptr1: *const T,
    ptr2: *const T,
}

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<MultiplePointers<U>> for MultiplePointers<T>
where
    T: Unsize<U>,
{} //~^^^ ERROR [E0378]


struct NothingToCoerce<T: ?Sized> {
    data: PhantomData<T>,
}

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<NothingToCoerce<T>> for NothingToCoerce<U> {}
//~^ ERROR [E0378]

#[repr(C)]
struct HasReprC<T: ?Sized>(Box<T>);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<HasReprC<U>> for HasReprC<T>
where
    T: Unsize<U>,
{} //~^^^ ERROR [E0378]

fn main() {}
