#![warn(clippy::implied_bounds_in_impl)]
#![allow(dead_code)]

use std::ops::{Deref, DerefMut};

trait Trait1<T> {}
// T is intentionally at a different position in Trait2 than in Trait1,
// since that also needs to be taken into account when making this lint work with generics
trait Trait2<U, T>: Trait1<T> {}
impl Trait1<i32> for () {}
impl Trait1<String> for () {}
impl Trait2<u32, i32> for () {}
impl Trait2<u32, String> for () {}

// Deref implied by DerefMut
fn deref_derefmut<T>(x: T) -> impl Deref<Target = T> + DerefMut<Target = T> {
    Box::new(x)
}

// Note: no test for different associated types needed since that isn't allowed in the first place.
// E.g. `-> impl Deref<Target = T> + DerefMut<Target = U>` is a compile error.

// DefIds of the traits match, but the generics do not, so it's *not* redundant.
// `Trait2: Trait` holds, but not `Trait2<_, String>: Trait1<i32>`.
// (Generic traits are currently not linted anyway but once/if ever implemented this should not
// warn.)
fn different_generics() -> impl Trait1<i32> + Trait2<u32, String> {
    /* () */
}

trait NonGenericTrait1 {}
trait NonGenericTrait2: NonGenericTrait1 {}
impl NonGenericTrait1 for i32 {}
impl NonGenericTrait2 for i32 {}

// Only one bound. Nothing to lint.
fn normal1() -> impl NonGenericTrait1 {
    1
}

fn normal2() -> impl NonGenericTrait1 + NonGenericTrait2 {
    1
}

fn main() {}
