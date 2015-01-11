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

#[on_unimplemented = "test error `{Self}` with `{Bar}` `{Baz}` `{Quux}`"]
trait Foo<Bar, Baz, Quux>{}

fn foobar<U: Clone, T: Foo<u8, U, u32>>() -> T {

}

#[on_unimplemented="a collection of type `{Self}` cannot be built from an iterator over elements of type `{A}`"]
trait MyFromIterator<A> {
    /// Build a container with elements from an external iterator.
    fn my_from_iter<T: Iterator<Item=A>>(iterator: T) -> Self;
}

fn collect<A, I: Iterator<Item=A>, B: MyFromIterator<A>>(it: I) -> B {
    MyFromIterator::my_from_iter(it)
}

#[on_unimplemented] //~ ERROR the #[on_unimplemented] attribute on trait definition for BadAnnotation1 must have a value, eg `#[on_unimplemented = "foo"]`
trait BadAnnotation1 {}

#[on_unimplemented = "Unimplemented trait error on `{Self}` with params `<{A},{B},{C}>`"]
//~^ ERROR the #[on_unimplemented] attribute on trait definition for BadAnnotation2<A, B> refers to non-existent type parameter C
trait BadAnnotation2<A,B> {}

fn trigger1<T: BadAnnotation1>(t: T)  {}
fn trigger2<A, B, T: BadAnnotation2<A,B>>(t: T) {}

pub fn main() {
    let x = vec!(1u8, 2, 3, 4);
    let y: Option<Vec<u8>> = collect(x.iter()); // this should give approximately the same error for x.iter().collect()
    //~^ ERROR
    //~^^ NOTE a collection of type `core::option::Option<collections::vec::Vec<u8>>` cannot be built from an iterator over elements of type `&u8`
    let x: String = foobar(); //~ ERROR
    //~^ NOTE test error `collections::string::String` with `u8` `_` `u32`

    // The following two have errors in their annotations, so the regular error should be thrown
    trigger1(1u8); //~ ERROR the trait `BadAnnotation1` is not implemented for the type `u8`
    trigger2::<u8, u8, u8>(1u8); //~ ERROR the trait `BadAnnotation2<u8, u8>` is not implemented

}
