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

#[repr(align(64))]
struct OverAlignedZst;
struct OverAligned<T: ?Sized>(Box<T>, OverAlignedZst);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<OverAligned<U>> for OverAligned<T>
    where
        T: Unsize<U>,
{} //~^^^ ERROR [E0378]

fn main() {}
