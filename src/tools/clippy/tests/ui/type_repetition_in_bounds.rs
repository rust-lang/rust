#![deny(clippy::type_repetition_in_bounds)]
#![allow(clippy::extra_unused_type_parameters)]

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub fn foo<T>(_t: T)
where
    T: Copy,
    T: Clone,
{
    unimplemented!();
}

pub fn bar<T, U>(_t: T, _u: U)
where
    T: Copy,
    U: Clone,
{
    unimplemented!();
}

// Threshold test (see #4380)
trait LintBounds
where
    Self: Clone,
    Self: Copy + Default + Ord,
    Self: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign,
{
}

trait LotsOfBounds
where
    Self: Clone + Copy + Default + Ord,
    Self: Add<Output = Self> + AddAssign + Sub<Output = Self> + SubAssign,
    Self: Mul<Output = Self> + MulAssign + Div<Output = Self> + DivAssign,
{
}

// Generic distinction (see #4323)
mod issue4323 {
    pub struct Foo<A>(A);
    pub struct Bar<A, B> {
        a: Foo<A>,
        b: Foo<B>,
    }

    impl<A, B> Unpin for Bar<A, B>
    where
        Foo<A>: Unpin,
        Foo<B>: Unpin,
    {
    }
}

// Extern macros shouldn't lint (see #4326)
extern crate serde;
mod issue4326 {
    use serde::{Deserialize, Serialize};

    trait Foo {}
    impl Foo for String {}

    #[derive(Debug, Serialize, Deserialize)]
    struct Bar<S>
    where
        S: Foo,
    {
        foo: S,
    }
}

// Issue #7360
struct Foo<T, U>
where
    T: Clone,
    U: Clone,
{
    t: T,
    u: U,
}

// Check for the `?` in `?Sized`
pub fn f<T: ?Sized>()
where
    T: Clone,
{
}
pub fn g<T: Clone>()
where
    T: ?Sized,
{
}

// This should not lint
fn impl_trait(_: impl AsRef<str>, _: impl AsRef<str>) {}

fn main() {}
