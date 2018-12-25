use std::borrow::{Cow, ToOwned};
use std::default::Default;
use std::iter::FromIterator;
use std::ops::Add;
use std::option::IntoIter as OptionIter;

pub struct XorShiftRng;
use XorShiftRng as DummyRng;
impl Rng for XorShiftRng {}
pub trait Rng {}
pub trait Rand: Default + Sized {
    fn rand<R: Rng>(_rng: &mut R) -> Self { Default::default() }
}
impl Rand for i32 { }

pub trait IntoCow<'a, B: ?Sized> where B: ToOwned {
    fn into_cow(self) -> Cow<'a, B>;
}

impl<'a> IntoCow<'a, str> for String {
    fn into_cow(self) -> Cow<'a, str> {
        Cow::Owned(self)
    }
}

#[derive(PartialEq, Eq)]
struct Newt<T>(T);

fn id<T>(x: T) -> T { x }
fn eq<T: Eq>(a: T, b: T) -> bool { a == b }
fn u8_as_i8(x: u8) -> i8 { x as i8 }
fn odd(x: usize) -> bool { x % 2 == 1 }
fn dummy_rng() -> DummyRng { XorShiftRng }

trait Size: Sized {
    fn size() -> usize { std::mem::size_of::<Self>() }
}
impl<T> Size for T {}

#[derive(PartialEq, Eq)]
struct BitVec;

impl BitVec {
    fn from_fn<F>(_: usize, _: F) -> BitVec where F: FnMut(usize) -> bool {
        BitVec
    }
}

#[derive(PartialEq, Eq)]
struct Foo<T>(T);

impl<T> Foo<T> {
    fn map_in_place<U, F>(self, mut f: F) -> Foo<U> where F: FnMut(T) -> U {
        Foo(f(self.0))
    }

}

macro_rules! tests {
    ($($expr:expr, $ty:ty, ($($test:expr),*);)+) => (pub fn main() {$({
        const C: $ty = $expr;
        static S: $ty = $expr;
        assert!(eq(C($($test),*), $expr($($test),*)));
        assert!(eq(S($($test),*), $expr($($test),*)));
        assert!(eq(C($($test),*), S($($test),*)));
    })+})
}

tests! {
    // Free function.
    id, fn(i32) -> i32, (5);
    id::<i32>, fn(i32) -> i32, (5);

    // Enum variant constructor.
    Some, fn(i32) -> Option<i32>, (5);
    Some::<i32>, fn(i32) -> Option<i32>, (5);

    // Tuple struct constructor.
    Newt, fn(i32) -> Newt<i32>, (5);
    Newt::<i32>, fn(i32) -> Newt<i32>, (5);

    // Inherent static methods.
    Vec::new, fn() -> Vec<()>, ();
    Vec::<()>::new, fn() -> Vec<()>, ();
    <Vec<()>>::new, fn() -> Vec<()>, ();
    Vec::with_capacity, fn(usize) -> Vec<()>, (5);
    Vec::<()>::with_capacity, fn(usize) -> Vec<()>, (5);
    <Vec<()>>::with_capacity, fn(usize) -> Vec<()>, (5);
    BitVec::from_fn, fn(usize, fn(usize) -> bool) -> BitVec, (5, odd);
    BitVec::from_fn::<fn(usize) -> bool>, fn(usize, fn(usize) -> bool) -> BitVec, (5, odd);

    // Inherent non-static method.
    Foo::map_in_place, fn(Foo<u8>, fn(u8) -> i8) -> Foo<i8>, (Foo(b'f'), u8_as_i8);
    Foo::map_in_place::<i8, fn(u8) -> i8>, fn(Foo<u8>, fn(u8) -> i8) -> Foo<i8>,
        (Foo(b'f'), u8_as_i8);
    Foo::<u8>::map_in_place, fn(Foo<u8>, fn(u8) -> i8) -> Foo<i8>
        , (Foo(b'f'), u8_as_i8);
    Foo::<u8>::map_in_place::<i8, fn(u8) -> i8>, fn(Foo<u8>, fn(u8) -> i8) -> Foo<i8>
        , (Foo(b'f'), u8_as_i8);

    // Trait static methods.
    bool::size, fn() -> usize, ();
    <bool>::size, fn() -> usize, ();
    <bool as Size>::size, fn() -> usize, ();

    Default::default, fn() -> i32, ();
    i32::default, fn() -> i32, ();
    <i32>::default, fn() -> i32, ();
    <i32 as Default>::default, fn() -> i32, ();

    Rand::rand, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    i32::rand, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    <i32>::rand, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    <i32 as Rand>::rand, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    Rand::rand::<DummyRng>, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    i32::rand::<DummyRng>, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    <i32>::rand::<DummyRng>, fn(&mut DummyRng) -> i32, (&mut dummy_rng());
    <i32 as Rand>::rand::<DummyRng>, fn(&mut DummyRng) -> i32, (&mut dummy_rng());

    // Trait non-static methods.
    Clone::clone, fn(&i32) -> i32, (&5);
    i32::clone, fn(&i32) -> i32, (&5);
    <i32>::clone, fn(&i32) -> i32, (&5);
    <i32 as Clone>::clone, fn(&i32) -> i32, (&5);

    FromIterator::from_iter, fn(OptionIter<i32>) -> Vec<i32>, (Some(5).into_iter());
    Vec::from_iter, fn(OptionIter<i32>) -> Vec<i32>, (Some(5).into_iter());
    <Vec<_>>::from_iter, fn(OptionIter<i32>) -> Vec<i32>, (Some(5).into_iter());
    <Vec<_> as FromIterator<_>>::from_iter, fn(OptionIter<i32>) -> Vec<i32>,
        (Some(5).into_iter());
    <Vec<i32> as FromIterator<_>>::from_iter, fn(OptionIter<i32>) -> Vec<i32>,
        (Some(5).into_iter());
    FromIterator::from_iter::<OptionIter<i32>>, fn(OptionIter<i32>) -> Vec<i32>,
        (Some(5).into_iter());
    <Vec<i32> as FromIterator<_>>::from_iter::<OptionIter<i32>>, fn(OptionIter<i32>) -> Vec<i32>,
        (Some(5).into_iter());

    Add::add, fn(i32, i32) -> i32, (5, 6);
    i32::add, fn(i32, i32) -> i32, (5, 6);
    <i32>::add, fn(i32, i32) -> i32, (5, 6);
    <i32 as Add<_>>::add, fn(i32, i32) -> i32, (5, 6);
    <i32 as Add<i32>>::add, fn(i32, i32) -> i32, (5, 6);

    String::into_cow, fn(String) -> Cow<'static, str>,
        ("foo".to_string());
    <String>::into_cow, fn(String) -> Cow<'static, str>,
        ("foo".to_string());
    <String as IntoCow<_>>::into_cow, fn(String) -> Cow<'static, str>,
        ("foo".to_string());
    <String as IntoCow<'static, _>>::into_cow, fn(String) -> Cow<'static, str>,
        ("foo".to_string());
}
