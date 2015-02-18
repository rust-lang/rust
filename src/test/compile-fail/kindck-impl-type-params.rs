// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #14061: tests the interaction between generic implementation
// parameter bounds and trait objects.

#![feature(box_syntax)]

use std::marker;

struct S<T>(marker::PhantomData<T>);

trait Gettable<T> {
    fn get(&self) -> T { panic!() }
}

impl<T: Send + Copy + 'static> Gettable<T> for S<T> {}

fn f<T>(val: T) {
    let t: S<T> = S(marker::PhantomData);
    let a = &t as &Gettable<T>;
    //~^ ERROR the trait `core::marker::Send` is not implemented
    //~^^ ERROR the trait `core::marker::Copy` is not implemented
}

fn g<T>(val: T) {
    let t: S<T> = S(marker::PhantomData);
    let a: &Gettable<T> = &t;
    //~^ ERROR the trait `core::marker::Send` is not implemented
    //~^^ ERROR the trait `core::marker::Copy` is not implemented
}

fn foo<'a>() {
    let t: S<&'a isize> = S(marker::PhantomData);
    let a = &t as &Gettable<&'a isize>;
    //~^ ERROR cannot infer
}

fn foo2<'a>() {
    let t: Box<S<String>> = box S(marker::PhantomData);
    let a = t as Box<Gettable<String>>;
    //~^ ERROR the trait `core::marker::Copy` is not implemented
}

fn foo3<'a>() {
    let t: Box<S<String>> = box S(marker::PhantomData);
    let a: Box<Gettable<String>> = t;
    //~^ ERROR the trait `core::marker::Copy` is not implemented
}

fn main() { }
