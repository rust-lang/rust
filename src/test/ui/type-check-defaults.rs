// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// compile-flags: --error-format=human

use std::iter::FromIterator;
use std::vec::IntoIter;
use std::ops::Add;

struct Foo<T, U: FromIterator<T>>(T, U);
struct WellFormed<Z = Foo<i32, i32>>(Z);
struct WellFormedNoBounds<Z:?Sized = Foo<i32, i32>>(Z);

struct WellFormedProjection<A, T=<A as Iterator>::Item>(A, T);

struct Bounds<T:Copy=String>(T);

struct WhereClause<T=String>(T) where T: Copy;

trait TraitBound<T:Copy=String> {}

trait SelfBound<T:Copy=Self> {}

trait FooTrait<T:Iterator = IntoIter<i32>> where T::Item : Add<u8> {}

trait Trait {}
struct TwoParams<T, U>(T, U);
impl Trait for TwoParams<i32, i32> {}
// Check that each default is substituted individually in the clauses.
struct Bogus<T = i32, U = i32>(TwoParams<T, U>) where TwoParams<T, U>: Trait;

fn main() { }
