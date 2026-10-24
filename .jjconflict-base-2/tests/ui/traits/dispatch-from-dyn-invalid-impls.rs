//! Test various invalid implementations of DispatchFromDyn trait.
//!
//! DispatchFromDyn is a special trait used by the compiler for dyn-compatible dynamic dispatch.
//! This checks that the compiler correctly rejects invalid implementations:
//! - Structs with extra non-coercible fields
//! - Structs with multiple pointer fields
//! - Structs with no coercible fields
//! - Structs with repr(C) or other incompatible representations
//! - Structs with over-aligned fields

#![feature(unsize, dispatch_from_dyn)]

use std::marker::{PhantomData, Unsize};
use std::ops::DispatchFromDyn;

// Extra field prevents DispatchFromDyn
struct WrapperWithExtraField<T>(T, i32);

impl<T, U> DispatchFromDyn<WrapperWithExtraField<U>> for WrapperWithExtraField<T>
//~^ ERROR [E0378]
where
    T: DispatchFromDyn<U>
{
}

// Multiple pointer fields create ambiguous coercion
struct MultiplePointers<T: ?Sized> {
    ptr1: *const T,
    ptr2: *const T,
}

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<MultiplePointers<U>> for MultiplePointers<T>
//~^ ERROR implementing `DispatchFromDyn` does not allow multiple fields to be coerced
where
    T: Unsize<U>
{
}

// No coercible fields (only PhantomData)
struct NothingToCoerce<T: ?Sized> {
    data: PhantomData<T>,
}

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<NothingToCoerce<T>> for NothingToCoerce<U> {}
//~^ ERROR implementing `DispatchFromDyn` requires a field to be coerced

// repr(C) is incompatible with DispatchFromDyn
#[repr(C)]
struct HasReprC<T: ?Sized>(Box<T>);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<HasReprC<U>> for HasReprC<T>
//~^ ERROR [E0378]
where
    T: Unsize<U>
{
}

// Over-aligned fields are incompatible
#[repr(align(64))]
struct OverAlignedZst;

struct OverAligned<T: ?Sized>(Box<T>, OverAlignedZst);

impl<T: ?Sized, U: ?Sized> DispatchFromDyn<OverAligned<U>> for OverAligned<T>
//~^ ERROR [E0378]
where
    T: Unsize<U>
{
}

fn main() {}
