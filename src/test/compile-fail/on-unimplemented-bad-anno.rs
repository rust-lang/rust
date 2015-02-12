// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
// ignore-tidy-linelength

#![feature(on_unimplemented)]

#![allow(unused)]

use std::marker;

#[rustc_on_unimplemented = "test error `{Self}` with `{Bar}` `{Baz}` `{Quux}`"]
trait Foo<Bar, Baz, Quux>
    : marker::PhantomFn<(Self,Bar,Baz,Quux)>
{}

#[rustc_on_unimplemented="a collection of type `{Self}` cannot be built from an iterator over elements of type `{A}`"]
trait MyFromIterator<A> {
    /// Build a container with elements from an external iterator.
    fn my_from_iter<T: Iterator<Item=A>>(iterator: T) -> Self;
}

#[rustc_on_unimplemented] //~ ERROR this attribute must have a value
trait BadAnnotation1
    : marker::MarkerTrait
{}

#[rustc_on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{C}>`"]
//~^ ERROR there is no type parameter C on trait BadAnnotation2
trait BadAnnotation2<A,B>
    : marker::PhantomFn<(Self,A,B)>
{}

#[rustc_on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{}>`"]
//~^ only named substitution parameters are allowed
trait BadAnnotation3<A,B>
    : marker::PhantomFn<(Self,A,B)>
{}

pub fn main() {
}
