#![deny(clippy::trait_duplication_in_bounds)]
#![allow(clippy::impl_trait_param)]

use std::collections::BTreeMap;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

fn bad_foo<T: Clone + Default, Z: Copy>(arg0: T, arg1: Z)
where
    T: Clone,
    T: Default,
{
    unimplemented!();
}

fn good_bar<T: Clone + Default>(arg: T) {
    unimplemented!();
}

fn good_foo<T>(arg: T)
where
    T: Clone + Default,
{
    unimplemented!();
}

fn good_foobar<T: Default>(arg: T)
where
    T: Clone,
{
    unimplemented!();
}

trait T: Default {
    fn f()
    where
        Self: Default;
}

trait U: Default {
    fn f()
    where
        Self: Clone;
}

trait ZZ: Default {
    fn g();
    fn h();
    fn f()
    where
        Self: Default + Clone;
}

trait BadTrait: Default + Clone {
    fn f()
    where
        Self: Default + Clone;
    fn g()
    where
        Self: Default;
    fn h()
    where
        Self: Copy;
}

#[derive(Default, Clone)]
struct Life;

impl T for Life {
    // this should not warn
    fn f() {}
}

impl U for Life {
    // this should not warn
    fn f() {}
}

// should not warn
trait Iter: Iterator {
    fn into_group_btreemap<K, V>(self) -> BTreeMap<K, Vec<V>>
    where
        Self: Iterator<Item = (K, V)> + Sized,
        K: Ord + Eq,
    {
        unimplemented!();
    }
}

struct Foo;

trait FooIter: Iterator<Item = Foo> {
    fn bar()
    where
        Self: Iterator<Item = Foo>,
    {
    }
}

// The below should not lint and exist to guard against false positives
fn impl_trait(_: impl AsRef<str>, _: impl AsRef<str>) {}

pub mod one {
    #[derive(Clone, Debug)]
    struct MultiProductIter<I>
    where
        I: Iterator + Clone,
        I::Item: Clone,
    {
        _marker: I,
    }

    pub struct MultiProduct<I>(Vec<MultiProductIter<I>>)
    where
        I: Iterator + Clone,
        I::Item: Clone;

    pub fn multi_cartesian_product<H>(_: H) -> MultiProduct<<H::Item as IntoIterator>::IntoIter>
    where
        H: Iterator,
        H::Item: IntoIterator,
        <H::Item as IntoIterator>::IntoIter: Clone,
        <H::Item as IntoIterator>::Item: Clone,
    {
        todo!()
    }
}

pub mod two {
    use std::iter::Peekable;

    pub struct MergeBy<I, J, F>
    where
        I: Iterator,
        J: Iterator<Item = I::Item>,
    {
        _i: Peekable<I>,
        _j: Peekable<J>,
        _f: F,
    }

    impl<I, J, F> Clone for MergeBy<I, J, F>
    where
        I: Iterator,
        J: Iterator<Item = I::Item>,
        std::iter::Peekable<I>: Clone,
        std::iter::Peekable<J>: Clone,
        F: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                _i: self._i.clone(),
                _j: self._j.clone(),
                _f: self._f.clone(),
            }
        }
    }
}

pub trait Trait {}

pub fn f(_a: impl Trait, _b: impl Trait) {}

pub trait ImplTrait<T> {}

impl<A, B> ImplTrait<(A, B)> for Foo where Foo: ImplTrait<A> + ImplTrait<B> {}

fn main() {}
