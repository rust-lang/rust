// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant)]
#![deny(visible_private_types)]
#![allow(dead_code)]
#![crate_type="lib"]

struct Private<T>;
pub struct Public<T>;

impl Private<Public<int>> {
    pub fn a(&self) -> Private<int> { fail!() }
    fn b(&self) -> Private<int> { fail!() }

    pub fn c() -> Private<int> { fail!() }
    fn d() -> Private<int> { fail!() }
}
impl Private<int> {
    pub fn e(&self) -> Private<int> { fail!() }
    fn f(&self) -> Private<int> { fail!() }
}

impl Public<Private<int>> {
    pub fn a(&self) -> Private<int> { fail!() }
    fn b(&self) -> Private<int> { fail!() }

    pub fn c() -> Private<int> { fail!() } //~ ERROR private type in exported type signature
    fn d() -> Private<int> { fail!() }
}
impl Public<int> {
    pub fn e(&self) -> Private<int> { fail!() } //~ ERROR private type in exported type signature
    fn f(&self) -> Private<int> { fail!() }
}

pub fn x(_: Private<int>) {} //~ ERROR private type in exported type signature

fn y(_: Private<int>) {}


pub struct Foo {
    pub x: Private<int>, //~ ERROR private type in exported type signature
    y: Private<int>
}

struct Bar {
    x: Private<int>,
}

pub enum Baz {
    Baz1(Private<int>), //~ ERROR private type in exported type signature
    Baz2 {
        pub x: Private<int>, //~ ERROR private type in exported type signature
        y: Private<int>
    },
}

enum Qux {
    Qux1(Private<int>),
    Qux2 {
        x: Private<int>,
    }
}

pub trait PubTrait {
    fn foo(&self) -> Private<int> { fail!( )} //~ ERROR private type in exported type signature
    fn bar(&self) -> Private<int>; //~ ERROR private type in exported type signature
    fn baz() -> Private<int>; //~ ERROR private type in exported type signature
}

impl PubTrait for Public<int> {
    fn bar(&self) -> Private<int> { fail!() }
    fn baz() -> Private<int> { fail!() }
}
impl PubTrait for Public<Private<int>> {
    fn bar(&self) -> Private<int> { fail!() }
    fn baz() -> Private<int> { fail!() }
}

impl PubTrait for Private<int> {
    fn bar(&self) -> Private<int> { fail!() }
    fn baz() -> Private<int> { fail!() }
}
impl PubTrait for (Private<int>,) {
    fn bar(&self) -> Private<int> { fail!() }
    fn baz() -> Private<int> { fail!() }
}


trait PrivTrait {
    fn foo(&self) -> Private<int> { fail!( )}
    fn bar(&self) -> Private<int>;
}
impl PrivTrait for Private<int> {
    fn bar(&self) -> Private<int> { fail!() }
}
impl PrivTrait for (Private<int>,) {
    fn bar(&self) -> Private<int> { fail!() }
}

pub trait ParamTrait<T> {
    fn foo() -> T;
}

impl ParamTrait<Private<int>> //~ ERROR private type in exported type signature
   for Public<int> {
    fn foo() -> Private<int> { fail!() }
}

impl ParamTrait<Private<int>> for Private<int> {
    fn foo() -> Private<int> { fail!( )}
}

impl<T: ParamTrait<Private<int>>>  //~ ERROR private type in exported type signature
     ParamTrait<T> for Public<i8> {
    fn foo() -> T { fail!() }
}
