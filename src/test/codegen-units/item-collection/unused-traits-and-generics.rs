// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=eager

#![crate_type="lib"]
#![deny(dead_code)]

// This test asserts that no codegen items are generated for generic items that
// are never instantiated in the local crate.

pub trait Trait {
    fn foo() {}
    fn bar(&self) {}
}

pub fn foo<T: Copy>(x: T) -> (T, T) {
    (x, x)
}

pub struct Struct<T> {
    x: T
}

impl<T> Struct<T> {
    pub fn foo(self) -> T {
        self.x
    }

    pub fn bar() {}
}

pub enum Enum<T> {
    A(T),
    B { x: T }
}

impl<T> Enum<T> {
    pub fn foo(self) -> T {
        match self {
            Enum::A(x) => x,
            Enum::B { x } => x,
        }
    }

    pub fn bar() {}
}

pub struct TupleStruct<T>(T);

impl<T> TupleStruct<T> {
    pub fn foo(self) -> T {
        self.0
    }

    pub fn bar() {}
}

pub type Pair<T> = (T, T);

pub struct NonGeneric {
    x: i32
}

impl NonGeneric {
    pub fn foo(self) -> i32 {
        self.x
    }

    pub fn generic_foo<T>(&self, x: T) -> (T, i32) {
        (x, self.x)
    }

    pub fn generic_bar<T: Copy>(x: T) -> (T, T) {
        (x, x)
    }
}

// Only the non-generic methods should be instantiated:
//~ TRANS_ITEM fn unused_traits_and_generics::{{impl}}[3]::foo[0]
//~ TRANS_ITEM drop-glue i8
