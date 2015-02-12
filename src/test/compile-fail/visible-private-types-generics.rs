// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo {
    fn dummy(&self) { }
}

pub fn f<
    T
    : Foo //~ ERROR private trait in exported type parameter bound
>() {}

pub fn g<T>() where
    T
    : Foo //~ ERROR private trait in exported type parameter bound
{}

pub struct S;

impl S {
    pub fn f<
        T
        : Foo //~ ERROR private trait in exported type parameter bound
    >() {}

    pub fn g<T>() where
        T
        : Foo //~ ERROR private trait in exported type parameter bound
    {}
}

pub struct S1<
    T
    : Foo //~ ERROR private trait in exported type parameter bound
> {
    x: T
}

pub struct S2<T> where
    T
    : Foo //~ ERROR private trait in exported type parameter bound
{
    x: T
}

pub enum E1<
    T
    : Foo //~ ERROR private trait in exported type parameter bound
> {
    V1(T)
}

pub enum E2<T> where
    T
    : Foo //~ ERROR private trait in exported type parameter bound
{
    V2(T)
}

fn main() {}
