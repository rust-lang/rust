#![feature(derive_smart_pointer, arbitrary_self_types)]

use std::marker::SmartPointer;

#[derive(SmartPointer)]
//~^ ERROR: `SmartPointer` can only be derived on `struct`s with `#[repr(transparent)]`
enum NotStruct<'a, T: ?Sized> {
    Variant(&'a T),
}

#[derive(SmartPointer)]
//~^ ERROR: At least one generic type should be designated as `#[pointee]` in order to derive `SmartPointer` traits
#[repr(transparent)]
struct NoPointee<'a, T: ?Sized> {
    ptr: &'a T,
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
//~^ ERROR: `SmartPointer` can only be derived on `struct`s with `#[repr(transparent)]`
struct NotTransparent<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

// However, reordering attributes should work nevertheless.
#[repr(transparent)]
#[derive(SmartPointer)]
struct ThisIsAPossibleSmartPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

fn main() {}
