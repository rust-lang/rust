// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! The Clone trait for types that cannot be "implicitly copied"

In Rust, some simple types are "implicitly copyable" and when you
assign them or pass them as arguments, the receiver will get a copy,
leaving the original value in place. These types do not require
allocation to copy and do not have finalizers (i.e. they do not
contain owned boxes or implement `Drop`), so the compiler considers
them cheap and safe to copy. For other types copies must be made
explicitly, by convention implementing the `Clone` trait and calling
the `clone` method.

*/

use std::kinds::Freeze;

/// A common trait for cloning an object.
pub trait Clone {
    /// Returns a copy of the value. The contents of owned pointers
    /// are copied to maintain uniqueness, while the contents of
    /// managed pointers are not copied.
    fn clone(&self) -> Self;

    /// Perform copy-assignment from `source`.
    ///
    /// `a.clone_from(&b)` is equivalent to `a = b.clone()` in functionality,
    /// but can be overriden to reuse the resources of `a` to avoid unnecessary
    /// allocations.
    #[inline(always)]
    fn clone_from(&mut self, source: &Self) {
        *self = source.clone()
    }
}

impl<T: Clone> Clone for ~T {
    /// Return a copy of the owned box.
    #[inline]
    fn clone(&self) -> ~T { ~(**self).clone() }

    /// Perform copy-assignment from `source` by reusing the existing allocation.
    fn clone_from(&mut self, source: &~T) {
        **self = (**source).clone()
    }
}

impl<T> Clone for @T {
    /// Return a shallow copy of the managed box.
    #[inline]
    fn clone(&self) -> @T { *self }
}

impl<T> Clone for @mut T {
    /// Return a shallow copy of the managed box.
    #[inline]
    fn clone(&self) -> @mut T { *self }
}

impl<'self, T> Clone for &'self T {
    /// Return a shallow copy of the borrowed pointer.
    #[inline]
    fn clone(&self) -> &'self T { *self }
}

impl<'self, T> Clone for &'self [T] {
    /// Return a shallow copy of the slice.
    #[inline]
    fn clone(&self) -> &'self [T] { *self }
}

impl<'self> Clone for &'self str {
    /// Return a shallow copy of the slice.
    #[inline]
    fn clone(&self) -> &'self str { *self }
}

macro_rules! clone_impl(
    ($t:ty) => {
        impl Clone for $t {
            /// Return a deep copy of the value.
            #[inline]
            fn clone(&self) -> $t { *self }
        }
    }
)

clone_impl!(int)
clone_impl!(i8)
clone_impl!(i16)
clone_impl!(i32)
clone_impl!(i64)

clone_impl!(uint)
clone_impl!(u8)
clone_impl!(u16)
clone_impl!(u32)
clone_impl!(u64)

clone_impl!(f32)
clone_impl!(f64)

clone_impl!(())
clone_impl!(bool)
clone_impl!(char)

macro_rules! extern_fn_clone(
    ($($A:ident),*) => (
        impl<$($A,)* ReturnType> Clone for extern "Rust" fn($($A),*) -> ReturnType {
            /// Return a copy of a function pointer
            #[inline]
            fn clone(&self) -> extern "Rust" fn($($A),*) -> ReturnType { *self }
        }
    )
)

extern_fn_clone!()
extern_fn_clone!(A)
extern_fn_clone!(A, B)
extern_fn_clone!(A, B, C)
extern_fn_clone!(A, B, C, D)
extern_fn_clone!(A, B, C, D, E)
extern_fn_clone!(A, B, C, D, E, F)
extern_fn_clone!(A, B, C, D, E, F, G)
extern_fn_clone!(A, B, C, D, E, F, G, H)

/// A trait distinct from `Clone` which represents "deep copies" of things like
/// managed boxes which would otherwise not be copied.
pub trait DeepClone: Clone {
    /// Return a deep copy of the value. Unlike `Clone`, the contents of shared pointer types
    /// *are* copied.
    fn deep_clone(&self) -> Self;

    /// Perform deep copy-assignment from `source`.
    ///
    /// `a.deep_clone_from(&b)` is equivalent to `a = b.deep_clone()` in
    /// functionality, but can be overriden to reuse the resources of `a` to
    /// avoid unnecessary allocations.
    #[inline(always)]
    fn deep_clone_from(&mut self, source: &Self) {
        *self = source.deep_clone()
    }
}

impl<T: DeepClone> DeepClone for ~T {
    /// Return a deep copy of the owned box.
    #[inline]
    fn deep_clone(&self) -> ~T { ~(**self).deep_clone() }

    /// Perform deep copy-assignment from `source` by reusing the existing allocation.
    fn deep_clone_from(&mut self, source: &~T) {
        **self = (**source).deep_clone()
    }
}

// FIXME: #6525: should also be implemented for `T: Send + DeepClone`
impl<T: Freeze + DeepClone + 'static> DeepClone for @T {
    /// Return a deep copy of the managed box. The `Freeze` trait is required to prevent performing
    /// a deep clone of a potentially cyclical type.
    #[inline]
    fn deep_clone(&self) -> @T { @(**self).deep_clone() }
}

// FIXME: #6525: should also be implemented for `T: Send + DeepClone`
impl<T: Freeze + DeepClone + 'static> DeepClone for @mut T {
    /// Return a deep copy of the managed box. The `Freeze` trait is required to prevent performing
    /// a deep clone of a potentially cyclical type.
    #[inline]
    fn deep_clone(&self) -> @mut T { @mut (**self).deep_clone() }
}

macro_rules! deep_clone_impl(
    ($t:ty) => {
        impl DeepClone for $t {
            /// Return a deep copy of the value.
            #[inline]
            fn deep_clone(&self) -> $t { *self }
        }
    }
)

deep_clone_impl!(int)
deep_clone_impl!(i8)
deep_clone_impl!(i16)
deep_clone_impl!(i32)
deep_clone_impl!(i64)

deep_clone_impl!(uint)
deep_clone_impl!(u8)
deep_clone_impl!(u16)
deep_clone_impl!(u32)
deep_clone_impl!(u64)

deep_clone_impl!(f32)
deep_clone_impl!(f64)

deep_clone_impl!(())
deep_clone_impl!(bool)
deep_clone_impl!(char)

macro_rules! extern_fn_deep_clone(
    ($($A:ident),*) => (
        impl<$($A,)* ReturnType> DeepClone for extern "Rust" fn($($A),*) -> ReturnType {
            /// Return a copy of a function pointer
            #[inline]
            fn deep_clone(&self) -> extern "Rust" fn($($A),*) -> ReturnType { *self }
        }
    )
)

extern_fn_deep_clone!()
extern_fn_deep_clone!(A)
extern_fn_deep_clone!(A, B)
extern_fn_deep_clone!(A, B, C)
extern_fn_deep_clone!(A, B, C, D)
extern_fn_deep_clone!(A, B, C, D, E)
extern_fn_deep_clone!(A, B, C, D, E, F)
extern_fn_deep_clone!(A, B, C, D, E, F, G)
extern_fn_deep_clone!(A, B, C, D, E, F, G, H)

#[test]
fn test_owned_clone() {
    let a = ~5i;
    let b: ~int = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_managed_clone() {
    let a = @5i;
    let b: @int = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_managed_mut_deep_clone() {
    let x = @mut 5i;
    let y: @mut int = x.deep_clone();
    *x = 20;
    assert_eq!(*y, 5);
}

#[test]
fn test_managed_mut_clone() {
    let a = @mut 5i;
    let b: @mut int = a.clone();
    assert_eq!(a, b);
    *b = 10;
    assert_eq!(a, b);
}

#[test]
fn test_borrowed_clone() {
    let x = 5i;
    let y: &int = &x;
    let z: &int = (&y).clone();
    assert_eq!(*z, 5);
}

#[test]
fn test_clone_from() {
    let a = ~5;
    let mut b = ~10;
    b.clone_from(&a);
    assert_eq!(*b, 5);
}

#[test]
fn test_deep_clone_from() {
    let a = ~5;
    let mut b = ~10;
    b.deep_clone_from(&a);
    assert_eq!(*b, 5);
}

#[test]
fn test_extern_fn_clone() {
    trait Empty {}
    impl Empty for int {}

    fn test_fn_a() -> f64 { 1.0 }
    fn test_fn_b<T: Empty>(x: T) -> T { x }
    fn test_fn_c(_: int, _: f64, _: ~[int], _: int, _: int, _: int) {}

    let _ = test_fn_a.clone();
    let _ = test_fn_b::<int>.clone();
    let _ = test_fn_c.clone();

    let _ = test_fn_a.deep_clone();
    let _ = test_fn_b::<int>.deep_clone();
    let _ = test_fn_c.deep_clone();
}
