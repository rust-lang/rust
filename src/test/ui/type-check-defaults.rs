// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::FromIterator;
use std::vec::IntoIter;
use std::ops::Add;

struct Foo<T, U: FromIterator<T>>(T, U);
struct WellFormed<Z = Foo<i32, i32>>(Z);
//~^ error: the trait bound `i32: std::iter::FromIterator<i32>` is not satisfied [E0277]
struct WellFormedNoBounds<Z:?Sized = Foo<i32, i32>>(Z);
//~^ error: the trait bound `i32: std::iter::FromIterator<i32>` is not satisfied [E0277]

struct WellFormedProjection<A, T=<A as Iterator>::Item>(A, T);
//~^ error: the trait bound `A: std::iter::Iterator` is not satisfied [E0277]

struct Bounds<T:Copy=String>(T);
//~^ error: the trait bound `std::string::String: std::marker::Copy` is not satisfied [E0277]

struct WhereClause<T=String>(T) where T: Copy;
//~^ error: the trait bound `std::string::String: std::marker::Copy` is not satisfied [E0277]

trait TraitBound<T:Copy=String> {}
//~^ error: the trait bound `std::string::String: std::marker::Copy` is not satisfied [E0277]

trait SelfBound<T:Copy=Self> {}
//~^ error: the trait bound `Self: std::marker::Copy` is not satisfied [E0277]

trait FooTrait<T:Iterator = IntoIter<i32>> where T::Item : Add<u8> {}
//~^ error: the trait bound `i32: std::ops::Add<u8>` is not satisfied [E0277]

trait Trait {}
struct TwoParams<T, U>(T, U);
impl Trait for TwoParams<i32, i32> {}
// Check that each default is substituted individually in the clauses.
struct Bogus<T = i32, U = i32>(TwoParams<T, U>) where TwoParams<T, U>: Trait;
//~^ error: the trait bound `TwoParams<i32, U>: Trait` is not satisfied [E0277]
//~^^ error: the trait bound `TwoParams<T, i32>: Trait` is not satisfied [E0277]

trait Super<T: Copy> { }
trait Base<T = String>: Super<T> { }
//~^ error: the trait bound `T: std::marker::Copy` is not satisfied [E0277]

fn main() { }
