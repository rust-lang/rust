#![feature(derive_coerce_pointee, arbitrary_self_types)]

extern crate core;
use std::marker::CoercePointee;

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with `#[repr(transparent)]`
enum NotStruct<'a, T: ?Sized> {
    Variant(&'a T),
}

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with at least one field
#[repr(transparent)]
struct NoField<'a, #[pointee] T: ?Sized> {}
//~^ ERROR: lifetime parameter `'a` is never used
//~| ERROR: type parameter `T` is never used

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with at least one field
#[repr(transparent)]
struct NoFieldUnit<'a, #[pointee] T: ?Sized>();
//~^ ERROR: lifetime parameter `'a` is never used
//~| ERROR: type parameter `T` is never used

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s that are generic over at least one type
#[repr(transparent)]
struct NoGeneric<'a>(&'a u8);

#[derive(CoercePointee)]
//~^ ERROR: exactly one generic type parameter must be marked as `#[pointee]` to derive `CoercePointee` traits
#[repr(transparent)]
struct AmbiguousPointee<'a, T1: ?Sized, T2: ?Sized> {
    a: (&'a T1, &'a T2),
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct TooManyPointees<'a, #[pointee] A: ?Sized, #[pointee] B: ?Sized>((&'a A, &'a B));
//~^ ERROR: only one type parameter can be marked as `#[pointee]` when deriving `CoercePointee` traits

#[derive(CoercePointee)]
//~^ ERROR: `CoercePointee` can only be derived on `struct`s with `#[repr(transparent)]`
struct NotTransparent<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct NoMaybeSized<'a, #[pointee] T> {
    //~^ ERROR: `derive(CoercePointee)` requires `T` to be marked `?Sized`
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeOnField<'a, #[pointee] T: ?Sized> {
    #[pointee]
    //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeInTypeConstBlock<
    'a,
    T: ?Sized = [u32; const {
                    struct UhOh<#[pointee] T>(T);
                    //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
                    10
                }],
> {
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeInConstConstBlock<
    'a,
    T: ?Sized,
    const V: u32 = {
        struct UhOh<#[pointee] T>(T);
        //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
        10
    },
> {
    ptr: &'a T,
}

#[derive(CoercePointee)]
#[repr(transparent)]
struct PointeeInAnotherTypeConstBlock<'a, #[pointee] T: ?Sized> {
    ptr: PointeeInConstConstBlock<
        'a,
        T,
        {
            struct UhOh<#[pointee] T>(T);
            //~^ ERROR: the `#[pointee]` attribute may only be used on generic parameters
            0
        },
    >,
}

// However, reordering attributes should work nevertheless.
#[repr(transparent)]
#[derive(CoercePointee)]
struct ThisIsAPossibleCoercePointee<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

// Also, these paths to Sized should work
#[derive(CoercePointee)]
#[repr(transparent)]
struct StdSized<'a, #[pointee] T: ?std::marker::Sized> {
    ptr: &'a T,
}
#[derive(CoercePointee)]
#[repr(transparent)]
struct CoreSized<'a, #[pointee] T: ?core::marker::Sized> {
    ptr: &'a T,
}
#[derive(CoercePointee)]
#[repr(transparent)]
struct GlobalStdSized<'a, #[pointee] T: ?::std::marker::Sized> {
    ptr: &'a T,
}
#[derive(CoercePointee)]
#[repr(transparent)]
struct GlobalCoreSized<'a, #[pointee] T: ?::core::marker::Sized> {
    ptr: &'a T,
}

fn main() {}
