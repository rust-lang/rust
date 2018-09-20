// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsize, coerce_sized, coerce_unsized)]

use std::{
    ops::{CoerceSized, CoerceUnsized},
    marker::{Unsize, PhantomData},
};

struct WrapperWithExtraField<T>(T, i32);

impl<T, U> CoerceUnsized<WrapperWithExtraField<U>> for WrapperWithExtraField<T>
where
    T: CoerceUnsized<U>,
{}

impl<T, U> CoerceSized<WrapperWithExtraField<T>> for WrapperWithExtraField<U>
where
    T: CoerceUnsized<U>,
    U: CoerceSized<T>,
{} //~^^^^ ERROR [E0378]


struct MultiplePointers<T: ?Sized>{
    ptr1: *const T,
    ptr2: *const T,
}

// No CoerceUnsized impl

impl<T: ?Sized, U: ?Sized> CoerceSized<MultiplePointers<T>> for MultiplePointers<U>
where
    T: Unsize<U>,
{} //~^^^ ERROR [E0378]


struct NothingToCoerce<T: ?Sized> {
    data: PhantomData<T>,
}

// No CoerceUnsized impl

impl<T: ?Sized, U: ?Sized> CoerceSized<NothingToCoerce<U>> for NothingToCoerce<T> {}
//~^ ERROR [E0378]

fn main() {}
