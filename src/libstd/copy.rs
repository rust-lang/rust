// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! The Copy trait for types that cannot be "implicitly copied"

In Rust, some simple types are "implicitly copyable" and when you
assign them or pass them as arguments, the receiver will get a copy,
leaving the original value in place. These types do not require
allocation to copy and do not have finalizers (i.e. they do not
contain owned boxes or implement `Drop`), so the compiler considers
them cheap and safe to copy. For other types copies must be made
explicitly, by convention implementing the `Copy` trait and calling
the `copy` method.

*/

use std::kinds::Freeze;

/// A common trait for copying an object.
pub trait Copy {
    /// Returns a copy of the value. The contents of owned pointers
    /// are copied to maintain uniqueness, while the contents of
    /// managed pointers are not copied.
    fn copy(&self) -> Self;
}

impl<T: Copy> Copy for ~T {
    /// Return a deep copy of the owned box.
    #[inline]
    fn copy(&self) -> ~T { ~(**self).copy() }
}

impl<T> Copy for @T {
    /// Return a shallow copy of the managed box.
    #[inline]
    fn copy(&self) -> @T { *self }
}

impl<T> Copy for @mut T {
    /// Return a shallow copy of the managed box.
    #[inline]
    fn copy(&self) -> @mut T { *self }
}

impl<'self, T> Copy for &'self T {
    /// Return a shallow copy of the borrowed pointer.
    #[inline]
    fn copy(&self) -> &'self T { *self }
}

impl<'self, T> Copy for &'self [T] {
    /// Return a shallow copy of the slice.
    #[inline]
    fn copy(&self) -> &'self [T] { *self }
}

impl<'self> Copy for &'self str {
    /// Return a shallow copy of the slice.
    #[inline]
    fn copy(&self) -> &'self str { *self }
}

macro_rules! copy_impl(
    ($t:ty) => {
        impl Copy for $t {
            /// Return a deep copy of the value.
            #[inline]
            fn copy(&self) -> $t { *self }
        }
    }
)

copy_impl!(int)
copy_impl!(i8)
copy_impl!(i16)
copy_impl!(i32)
copy_impl!(i64)

copy_impl!(uint)
copy_impl!(u8)
copy_impl!(u16)
copy_impl!(u32)
copy_impl!(u64)

copy_impl!(float)
copy_impl!(f32)
copy_impl!(f64)

copy_impl!(())
copy_impl!(bool)
copy_impl!(char)

macro_rules! extern_fn_copy(
    ($($A:ident),*) => (
        impl<$($A,)* ReturnType> Copy for extern "Rust" fn($($A),*) -> ReturnType {
            /// Return a copy of a function pointer
            #[inline]
            fn copy(&self) -> extern "Rust" fn($($A),*) -> ReturnType { *self }
        }
    )
)

extern_fn_copy!()
extern_fn_copy!(A)
extern_fn_copy!(A, B)
extern_fn_copy!(A, B, C)
extern_fn_copy!(A, B, C, D)
extern_fn_copy!(A, B, C, D, E)
extern_fn_copy!(A, B, C, D, E, F)
extern_fn_copy!(A, B, C, D, E, F, G)
extern_fn_copy!(A, B, C, D, E, F, G, H)

/// A trait distinct from `Copy` which represents "deep copies" of things like
/// managed boxes which would otherwise not be copied.
pub trait DeepCopy {
    /// Return a deep copy of the value. Unlike `Copy`, the contents of shared pointer types
    /// *are* copied.
    fn deep_copy(&self) -> Self;
}

impl<T: DeepCopy> DeepCopy for ~T {
    /// Return a deep copy of the owned box.
    #[inline]
    fn deep_copy(&self) -> ~T { ~(**self).deep_copy() }
}

// FIXME: #6525: should also be implemented for `T: Send + DeepCopy`
impl<T: Freeze + DeepCopy + 'static> DeepCopy for @T {
    /// Return a deep copy of the managed box. The `Freeze` trait is required to prevent performing
    /// a deep copy of a potentially cyclical type.
    #[inline]
    fn deep_copy(&self) -> @T { @(**self).deep_copy() }
}

// FIXME: #6525: should also be implemented for `T: Send + DeepCopy`
impl<T: Freeze + DeepCopy + 'static> DeepCopy for @mut T {
    /// Return a deep copy of the managed box. The `Freeze` trait is required to prevent performing
    /// a deep copy of a potentially cyclical type.
    #[inline]
    fn deep_copy(&self) -> @mut T { @mut (**self).deep_copy() }
}

macro_rules! deep_copy_impl(
    ($t:ty) => {
        impl DeepCopy for $t {
            /// Return a deep copy of the value.
            #[inline]
            fn deep_copy(&self) -> $t { *self }
        }
    }
)

deep_copy_impl!(int)
deep_copy_impl!(i8)
deep_copy_impl!(i16)
deep_copy_impl!(i32)
deep_copy_impl!(i64)

deep_copy_impl!(uint)
deep_copy_impl!(u8)
deep_copy_impl!(u16)
deep_copy_impl!(u32)
deep_copy_impl!(u64)

deep_copy_impl!(float)
deep_copy_impl!(f32)
deep_copy_impl!(f64)

deep_copy_impl!(())
deep_copy_impl!(bool)
deep_copy_impl!(char)

macro_rules! extern_fn_deep_copy(
    ($($A:ident),*) => (
        impl<$($A,)* ReturnType> DeepCopy for extern "Rust" fn($($A),*) -> ReturnType {
            /// Return a copy of a function pointer
            #[inline]
            fn deep_copy(&self) -> extern "Rust" fn($($A),*) -> ReturnType { *self }
        }
    )
)

extern_fn_deep_copy!()
extern_fn_deep_copy!(A)
extern_fn_deep_copy!(A, B)
extern_fn_deep_copy!(A, B, C)
extern_fn_deep_copy!(A, B, C, D)
extern_fn_deep_copy!(A, B, C, D, E)
extern_fn_deep_copy!(A, B, C, D, E, F)
extern_fn_deep_copy!(A, B, C, D, E, F, G)
extern_fn_deep_copy!(A, B, C, D, E, F, G, H)

#[test]
fn test_owned_copy() {
    let a = ~5i;
    let b: ~int = a.copy();
    assert_eq!(a, b);
}

#[test]
fn test_managed_copy() {
    let a = @5i;
    let b: @int = a.copy();
    assert_eq!(a, b);
}

#[test]
fn test_managed_mut_deep_copy() {
    let x = @mut 5i;
    let y: @mut int = x.deep_copy();
    *x = 20;
    assert_eq!(*y, 5);
}

#[test]
fn test_managed_mut_copy() {
    let a = @mut 5i;
    let b: @mut int = a.copy();
    assert_eq!(a, b);
    *b = 10;
    assert_eq!(a, b);
}

#[test]
fn test_borrowed_copy() {
    let x = 5i;
    let y: &int = &x;
    let z: &int = (&y).copy();
    assert_eq!(*z, 5);
}

#[test]
fn test_extern_fn_copy() {
    trait Empty {}
    impl Empty for int {}

    fn test_fn_a() -> float { 1.0 }
    fn test_fn_b<T: Empty>(x: T) -> T { x }
    fn test_fn_c(_: int, _: float, _: ~[int], _: int, _: int, _: int) {}

    let _ = test_fn_a.copy();
    let _ = test_fn_b::<int>.copy();
    let _ = test_fn_c.copy();

    let _ = test_fn_a.deep_copy();
    let _ = test_fn_b::<int>.deep_copy();
    let _ = test_fn_c.deep_copy();
}
