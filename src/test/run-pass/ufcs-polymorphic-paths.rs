// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(collections, rand, into_cow)]

use std::borrow::{Cow, IntoCow};
use std::collections::BitVec;
use std::default::Default;
use std::iter::FromIterator;
use std::ops::Add;
use std::option::IntoIter as OptionIter;
use std::rand::Rand;
use std::rand::XorShiftRng as DummyRng;
// FIXME the glob std::prelude::*; import of Vec is missing non-static inherent methods.
use std::vec::Vec;

#[derive(PartialEq, Eq)]
struct Newt<T>(T);

fn id<T>(x: T) -> T { x }
fn eq<T: Eq>(a: T, b: T) -> bool { a == b }
fn u8_as_i8(x: u8) -> i8 { x as i8 }
fn odd(x: usize) -> bool { x % 2 == 1 }
fn dummy_rng() -> DummyRng { DummyRng::new_unseeded() }

trait Size: Sized {
    fn size() -> usize { std::mem::size_of::<Self>() }
}
impl<T> Size for T {}

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
    Vec::map_in_place, fn(Vec<u8>, fn(u8) -> i8) -> Vec<i8>, (vec![b'f', b'o', b'o'], u8_as_i8);
    Vec::map_in_place::<i8, fn(u8) -> i8>, fn(Vec<u8>, fn(u8) -> i8) -> Vec<i8>,
        (vec![b'f', b'o', b'o'], u8_as_i8);
    // FIXME these break with "type parameter might not appear here pointing at `<u8>`.
    // Vec::<u8>::map_in_place: fn(Vec<u8>, fn(u8) -> i8) -> Vec<i8>
    //    , (vec![b'f', b'o', b'o'], u8_as_i8);
    // Vec::<u8>::map_in_place::<i8, fn(u8) -> i8>: fn(Vec<u8>, fn(u8) -> i8) -> Vec<i8>
    //    , (vec![b'f', b'o', b'o'], u8_as_i8);

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
