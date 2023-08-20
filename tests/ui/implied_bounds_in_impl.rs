#![warn(clippy::implied_bounds_in_impl)]
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
// U is intentionally at a different "index" in GenericSubtrait than `T` is in GenericTrait
trait GenericSubtrait<T, U, V>: GenericTrait<U> + GenericTrait2<V> {}

impl GenericTrait<i32> for () {}
impl GenericTrait<i64> for () {}
impl<V> GenericTrait2<V> for () {}
impl<V> GenericSubtrait<(), i32, V> for () {}
impl<V> GenericSubtrait<(), i64, V> for () {}

fn generics_implied<T>() -> impl GenericTrait<T> + GenericSubtrait<(), T, ()>
where
    (): GenericSubtrait<(), T, ()>,
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

fn main() {}
