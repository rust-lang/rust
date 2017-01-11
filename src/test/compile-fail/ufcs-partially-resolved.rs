// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_type_defaults)]

trait Tr {
    type Y = u16;
    fn Y() {}
}
impl Tr for u8 {}

trait Dr {
    type X = u16;
    fn Z() {}
}
impl Dr for u8 {}

enum E { Y }
type A = u32;

fn main() {
    let _: <u8 as Tr>::N; //~ ERROR unresolved associated type `Tr::N`
    let _: <u8 as E>::N; //~ ERROR unresolved associated type `E::N`
    let _: <u8 as A>::N; //~ ERROR unresolved associated type `A::N`
    <u8 as Tr>::N; //~ ERROR unresolved method or associated constant `Tr::N`
    <u8 as E>::N; //~ ERROR unresolved method or associated constant `E::N`
    <u8 as A>::N; //~ ERROR unresolved method or associated constant `A::N`
    let _: <u8 as Tr>::Y; // OK
    let _: <u8 as E>::Y; //~ ERROR expected associated type, found variant `E::Y`
    <u8 as Tr>::Y; // OK
    <u8 as E>::Y; //~ ERROR expected method or associated constant, found unit variant `E::Y`

    let _: <u8 as Tr>::N::NN; //~ ERROR unresolved associated type `Tr::N`
    let _: <u8 as E>::N::NN; //~ ERROR unresolved associated type `E::N`
    let _: <u8 as A>::N::NN; //~ ERROR unresolved associated type `A::N`
    <u8 as Tr>::N::NN; //~ ERROR unresolved associated type `Tr::N`
    <u8 as E>::N::NN; //~ ERROR unresolved associated type `E::N`
    <u8 as A>::N::NN; //~ ERROR unresolved associated type `A::N`
    let _: <u8 as Tr>::Y::NN; //~ ERROR ambiguous associated type
    let _: <u8 as E>::Y::NN; //~ ERROR expected associated type, found variant `E::Y`
    <u8 as Tr>::Y::NN; //~ ERROR no associated item named `NN` found for type `<u8 as Tr>::Y`
    <u8 as E>::Y::NN; //~ ERROR expected associated type, found variant `E::Y`

    let _: <u8 as Tr::N>::NN; //~ ERROR unresolved associated type `Tr::N::NN`
    let _: <u8 as E::N>::NN; //~ ERROR unresolved associated type `E::N::NN`
    let _: <u8 as A::N>::NN; //~ ERROR unresolved associated type `A::N::NN`
    <u8 as Tr::N>::NN; //~ ERROR unresolved method or associated constant `Tr::N::NN`
    <u8 as E::N>::NN; //~ ERROR unresolved method or associated constant `E::N::NN`
    <u8 as A::N>::NN; //~ ERROR unresolved method or associated constant `A::N::NN`
    let _: <u8 as Tr::Y>::NN; //~ ERROR unresolved associated type `Tr::Y::NN`
    let _: <u8 as E::Y>::NN; //~ ERROR failed to resolve. Not a module `Y`
    <u8 as Tr::Y>::NN; //~ ERROR unresolved method or associated constant `Tr::Y::NN`
    <u8 as E::Y>::NN; //~ ERROR failed to resolve. Not a module `Y`

    let _: <u8 as Dr>::Z; //~ ERROR expected associated type, found method `Dr::Z`
    <u8 as Dr>::X; //~ ERROR expected method or associated constant, found associated type `Dr::X`
    let _: <u8 as Dr>::Z::N; //~ ERROR expected associated type, found method `Dr::Z`
    <u8 as Dr>::X::N; //~ ERROR no associated item named `N` found for type `<u8 as Dr>::X`
}
