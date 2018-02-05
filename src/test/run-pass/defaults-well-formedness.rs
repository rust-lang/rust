// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait<T> {}
struct Foo<U, V=i32>(U, V) where U: Trait<V>;

trait Marker {}
struct TwoParams<T, U>(T, U);
impl Marker for TwoParams<i32, i32> {}
// Check that defaults are substituted simultaneously.
struct IndividuallyBogus<T = i32, U = i32>(TwoParams<T, U>) where TwoParams<T, U>: Marker;
// Clauses with non-defaulted params are not checked.
struct NonDefaultedInClause<T, U = i32>(TwoParams<T, U>) where TwoParams<T, U>: Marker;
struct DefaultedLhs<U, V=i32>(U, V) where V: Trait<U>;
// Dependent defaults.
struct Dependent<T: Copy, U = T>(T, U) where U: Copy;

fn main() {}
