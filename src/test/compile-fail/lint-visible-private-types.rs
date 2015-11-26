// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![crate_type="lib"]

use std::marker;

struct Private<T>(marker::PhantomData<T>);
pub struct Public<T>(marker::PhantomData<T>);

impl Private<Public<isize>> {
    pub fn a(&self) -> Private<isize> { panic!() }
    fn b(&self) -> Private<isize> { panic!() }

    pub fn c() -> Private<isize> { panic!() }
    fn d() -> Private<isize> { panic!() }
}
impl Private<isize> {
    pub fn e(&self) -> Private<isize> { panic!() }
    fn f(&self) -> Private<isize> { panic!() }
}

impl Public<Private<isize>> {
    pub fn a(&self) -> Private<isize> { panic!() }
    fn b(&self) -> Private<isize> { panic!() }

    pub fn c() -> Private<isize> { panic!() }
    fn d() -> Private<isize> { panic!() }
}
impl Public<isize> {
    pub fn e(&self) -> Private<isize> { panic!() } //~ ERROR private type in public interface
    fn f(&self) -> Private<isize> { panic!() }
}

pub fn x(_: Private<isize>) {} //~ ERROR private type in public interface

fn y(_: Private<isize>) {}


pub struct Foo {
    pub x: Private<isize>, //~ ERROR private type in public interface
    y: Private<isize>
}

struct Bar {
    x: Private<isize>,
}

pub enum Baz {
    Baz1(Private<isize>), //~ ERROR private type in public interface
    Baz2 {
        y: Private<isize> //~ ERROR private type in public interface
    },
}

enum Qux {
    Qux1(Private<isize>),
    Qux2 {
        x: Private<isize>,
    }
}

pub trait PubTrait {
    fn foo(&self) -> Private<isize> { panic!( )} //~ ERROR private type in public interface
    fn bar(&self) -> Private<isize>; //~ ERROR private type in public interface
    fn baz() -> Private<isize>; //~ ERROR private type in public interface
}

impl PubTrait for Public<isize> {
    fn bar(&self) -> Private<isize> { panic!() } // Warns in lint checking phase
    fn baz() -> Private<isize> { panic!() } // Warns in lint checking phase
}
impl PubTrait for Public<Private<isize>> {
    fn bar(&self) -> Private<isize> { panic!() }
    fn baz() -> Private<isize> { panic!() }
}

impl PubTrait for Private<isize> {
    fn bar(&self) -> Private<isize> { panic!() }
    fn baz() -> Private<isize> { panic!() }
}
impl PubTrait for (Private<isize>,) {
    fn bar(&self) -> Private<isize> { panic!() }
    fn baz() -> Private<isize> { panic!() }
}


trait PrivTrait {
    fn foo(&self) -> Private<isize> { panic!( )}
    fn bar(&self) -> Private<isize>;
}
impl PrivTrait for Private<isize> {
    fn bar(&self) -> Private<isize> { panic!() }
}
impl PrivTrait for (Private<isize>,) {
    fn bar(&self) -> Private<isize> { panic!() }
}

pub trait ParamTrait<T> {
    fn foo() -> T;
}

impl ParamTrait<Private<isize>>
   for Public<isize> {
    fn foo() -> Private<isize> { panic!() }
}

impl ParamTrait<Private<isize>> for Private<isize> {
    fn foo() -> Private<isize> { panic!( )}
}

impl<T: ParamTrait<Private<isize>>>  //~ ERROR private type in public interface
     ParamTrait<T> for Public<i8> {
    fn foo() -> T { panic!() }
}

type PrivAliasPrivType = Private<isize>;
pub fn f1(_: PrivAliasPrivType) {} //~ ERROR private type in public interface

type PrivAliasGeneric<T = Private<isize>> = T;
pub fn f2(_: PrivAliasGeneric) {} //~ ERROR private type in public interface

type Result<T> = std::result::Result<T, Private<isize>>;
pub fn f3(_: Result<u8>) {} //~ ERROR private type in public interface
