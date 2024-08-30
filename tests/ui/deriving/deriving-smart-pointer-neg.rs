#![feature(derive_smart_pointer, arbitrary_self_types)]

extern crate core;
use std::marker::SmartPointer;

#[derive(SmartPointer)]
//~^ ERROR: `SmartPointer` can only be derived on `struct`s with `#[repr(transparent)]`
enum NotStruct<'a, T: ?Sized> {
    Variant(&'a T),
}

#[derive(SmartPointer)]
//~^ ERROR: `SmartPointer` can only be derived on `struct`s with at least one field
#[repr(transparent)]
struct NoField<'a, #[pointee] T: ?Sized> {}
//~^ ERROR: lifetime parameter `'a` is never used
//~| ERROR: type parameter `T` is never used

#[derive(SmartPointer)]
//~^ ERROR: `SmartPointer` can only be derived on `struct`s with at least one field
#[repr(transparent)]
struct NoFieldUnit<'a, #[pointee] T: ?Sized>();
//~^ ERROR: lifetime parameter `'a` is never used
//~| ERROR: type parameter `T` is never used

#[derive(SmartPointer)]
//~^ ERROR: `SmartPointer` can only be derived on `struct`s that are generic over at least one type
#[repr(transparent)]
struct NoGeneric<'a>(&'a u8);

#[derive(SmartPointer)]
//~^ ERROR: exactly one generic type parameter must be marked as #[pointee] to derive SmartPointer traits
#[repr(transparent)]
struct AmbiguousPointee<'a, T1: ?Sized, T2: ?Sized> {
    a: (&'a T1, &'a T2),
}

#[derive(SmartPointer)]
#[repr(transparent)]
struct TooManyPointees<'a, #[pointee] A: ?Sized, #[pointee] B: ?Sized>((&'a A, &'a B));
//~^ ERROR: only one type parameter can be marked as `#[pointee]` when deriving SmartPointer traits

#[derive(SmartPointer)]
//~^ ERROR: `SmartPointer` can only be derived on `struct`s with `#[repr(transparent)]`
struct NotTransparent<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

#[derive(SmartPointer)]
#[repr(transparent)]
struct NoMaybeSized<'a, #[pointee] T> {
    //~^ ERROR: `derive(SmartPointer)` requires T to be marked `?Sized`
    ptr: &'a T,
}

// However, reordering attributes should work nevertheless.
#[repr(transparent)]
#[derive(SmartPointer)]
struct ThisIsAPossibleSmartPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

// Also, these paths to Sized should work
#[derive(SmartPointer)]
#[repr(transparent)]
struct StdSized<'a, #[pointee] T: ?std::marker::Sized> {
    ptr: &'a T,
}
#[derive(SmartPointer)]
#[repr(transparent)]
struct CoreSized<'a, #[pointee] T: ?core::marker::Sized> {
    ptr: &'a T,
}
#[derive(SmartPointer)]
#[repr(transparent)]
struct GlobalStdSized<'a, #[pointee] T: ?::std::marker::Sized> {
    ptr: &'a T,
}
#[derive(SmartPointer)]
#[repr(transparent)]
struct GlobalCoreSized<'a, #[pointee] T: ?::core::marker::Sized> {
    ptr: &'a T,
}

fn main() {}
