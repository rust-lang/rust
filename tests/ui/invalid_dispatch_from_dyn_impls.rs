#![feature(unsize, dispatch_from_dyn)]

use std::{
    ops::DispatchFromDyn,
    marker::{Unsize, PhantomData},
};

struct WrapperWithExtraField<T>(T, i32);

impl<T, U> DispatchFromDyn<WrapperWithExtraField<U>> for WrapperWithExtraField<T>
//~^ ERROR [E0378]
where
    T: DispatchFromDyn<U>,
{}


struct MultiplePointers<T: ?Sized>{
    ptr1: *const T,
    ptr2: *const T,
}

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<MultiplePointers<U>> for MultiplePointers<T>
//~^ ERROR implementing `DispatchFromDyn` does not allow multiple fields to be coerced
where
    T: Unsize<U>,
{}


struct NothingToCoerce<T: ?Sized> {
    data: PhantomData<T>,
}

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<NothingToCoerce<T>> for NothingToCoerce<U> {}
//~^ ERROR implementing `DispatchFromDyn` requires a field to be coerced

#[repr(C)]
struct HasReprC<T: ?Sized>(Box<T>);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<HasReprC<U>> for HasReprC<T>
//~^ ERROR [E0378]
where
    T: Unsize<U>,
{}

#[repr(align(64))]
struct OverAlignedZst;
struct OverAligned<T: ?Sized>(Box<T>, OverAlignedZst);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<OverAligned<U>> for OverAligned<T>
//~^ ERROR [E0378]
    where
        T: Unsize<U>,
{}

fn main() {}
