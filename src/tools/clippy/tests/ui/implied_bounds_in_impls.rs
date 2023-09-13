#![warn(clippy::implied_bounds_in_impls)]
#![allow(dead_code)]

use std::ops::{Deref, DerefMut};

// Only one bound, nothing to lint.
fn normal_deref<T>(x: T) -> impl Deref<Target = T> {
    Box::new(x)
}

// Deref implied by DerefMut
fn deref_derefmut<T>(x: T) -> impl Deref<Target = T> + DerefMut<Target = T> {
    Box::new(x)
}

trait GenericTrait<T> {}
trait GenericTrait2<V> {}
// U is intentionally at a different "index" in GenericSubtrait than `T` is in GenericTrait,
// so this can be a good test to make sure that the calculations are right (no off-by-one errors,
// ...)
trait GenericSubtrait<T, U, V>: GenericTrait<U> + GenericTrait2<V> {}

impl GenericTrait<i32> for () {}
impl GenericTrait<i64> for () {}
impl<V> GenericTrait2<V> for () {}
impl<V> GenericSubtrait<(), i32, V> for () {}
impl<V> GenericSubtrait<(), i64, V> for () {}

fn generics_implied<U, W>() -> impl GenericTrait<W> + GenericSubtrait<U, W, U>
where
    (): GenericSubtrait<U, W, U>,
{
}

fn generics_implied_multi<V>() -> impl GenericTrait<i32> + GenericTrait2<V> + GenericSubtrait<(), i32, V> {}

fn generics_implied_multi2<T, V>() -> impl GenericTrait<T> + GenericTrait2<V> + GenericSubtrait<(), T, V>
where
    (): GenericSubtrait<(), T, V> + GenericTrait<T>,
{
}

// i32 != i64, GenericSubtrait<_, i64, _> does not imply GenericTrait<i32>, don't lint
fn generics_different() -> impl GenericTrait<i32> + GenericSubtrait<(), i64, ()> {}

// i32 == i32, GenericSubtrait<_, i32, _> does imply GenericTrait<i32>, lint
fn generics_same() -> impl GenericTrait<i32> + GenericSubtrait<(), i32, ()> {}

trait SomeTrait {
    // Check that it works in trait declarations.
    fn f() -> impl Deref + DerefMut<Target = u8>;
}
struct SomeStruct;
impl SomeStruct {
    // Check that it works in inherent impl blocks.
    fn f() -> impl Deref + DerefMut<Target = u8> {
        Box::new(123)
    }
}
impl SomeTrait for SomeStruct {
    // Check that it works in trait impls.
    fn f() -> impl Deref + DerefMut<Target = u8> {
        Box::new(42)
    }
}

mod issue11422 {
    use core::fmt::Debug;
    // Some additional tests that would cause ICEs:

    // `PartialOrd` has a default generic parameter and does not need to be explicitly specified.
    // This needs special handling.
    fn default_generic_param1() -> impl PartialEq + PartialOrd + Debug {}
    fn default_generic_param2() -> impl PartialOrd + PartialEq + Debug {}

    // Referring to `Self` in the supertrait clause needs special handling.
    trait Trait1<X: ?Sized> {}
    trait Trait2: Trait1<Self> {}
    impl Trait1<()> for () {}
    impl Trait2 for () {}

    fn f() -> impl Trait1<()> + Trait2 {}
}

mod issue11435 {
    // Associated type needs to be included on DoubleEndedIterator in the suggestion
    fn my_iter() -> impl Iterator<Item = u32> + DoubleEndedIterator {
        0..5
    }

    // Removing the `Clone` bound should include the `+` behind it in its remove suggestion
    fn f() -> impl Copy + Clone {
        1
    }

    trait Trait1<T> {
        type U;
    }
    impl Trait1<i32> for () {
        type U = i64;
    }
    trait Trait2<T>: Trait1<T> {}
    impl Trait2<i32> for () {}

    // When the other trait has generics, it shouldn't add another pair of `<>`
    fn f2() -> impl Trait1<i32, U = i64> + Trait2<i32> {}

    trait Trait3<T, U, V> {
        type X;
        type Y;
    }
    trait Trait4<T>: Trait3<T, i16, i64> {}
    impl Trait3<i8, i16, i64> for () {
        type X = i32;
        type Y = i128;
    }
    impl Trait4<i8> for () {}

    // Associated type `X` is specified, but `Y` is not, so only that associated type should be moved
    // over
    fn f3() -> impl Trait3<i8, i16, i64, X = i32, Y = i128> + Trait4<i8, X = i32> {}
}

fn main() {}
