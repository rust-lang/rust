#![deny(clippy::type_repetition_in_bounds)]
#![allow(
    clippy::extra_unused_type_parameters,
    clippy::multiple_bound_locations,
    clippy::needless_maybe_sized
)]

use serde::Deserialize;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub fn foo<T>(_t: T)
where
    T: Copy,
    T: Clone,
    //~^ type_repetition_in_bounds
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
    //~^ type_repetition_in_bounds
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

// Extern macros shouldn't lint, again (see #10504)
mod issue10504 {
    use serde::{Deserialize, Serialize};
    use std::fmt::Debug;
    use std::hash::Hash;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(bound(
        serialize = "T: Serialize + Hash + Eq",
        deserialize = "Box<T>: serde::de::DeserializeOwned + Hash + Eq"
    ))]
    struct OpaqueParams<T: ?Sized + Debug>(std::marker::PhantomData<T>);
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
    //~^ type_repetition_in_bounds
{
}
pub fn g<T: Clone>()
where
    T: ?Sized,
    //~^ type_repetition_in_bounds
{
}

// This should not lint
fn impl_trait(_: impl AsRef<str>, _: impl AsRef<str>) {}

#[clippy::msrv = "1.14.0"]
mod issue8772_fail {
    pub trait Trait<X, Y, Z> {}

    pub fn f<T: ?Sized, U>(arg: usize)
    where
        T: Trait<Option<usize>, Box<[String]>, bool> + 'static,
        U: Clone + Sync + 'static,
    {
    }
}

#[clippy::msrv = "1.15.0"]
mod issue8772_pass {
    pub trait Trait<X, Y, Z> {}

    pub fn f<T: ?Sized, U>(arg: usize)
    where
        T: Trait<Option<usize>, Box<[String]>, bool> + 'static,
        //~^ type_repetition_in_bounds
        U: Clone + Sync + 'static,
    {
    }
}

struct Issue14744<'a, K: 'a>
where
    K: Clone,
{
    phantom: std::marker::PhantomData<&'a K>,
}
//~^^^^ type_repetition_in_bounds

struct ComplexType<T>
where
    Vec<T>: Clone,
    Vec<T>: Clone,
{
    t: T,
}
//~^^^^ type_repetition_in_bounds

fn main() {}
