// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Trait {}

struct Foo<T:Trait> {
    x: T,
}

enum Bar<T:Trait> {
    ABar(int),
    BBar(T),
    CBar(uint),
}

fn explode(x: Foo<u32>) {}
//~^ ERROR not implemented

fn kaboom(y: Bar<f32>) {}
//~^ ERROR not implemented

impl<T> Foo<T> {
//~^ ERROR the trait `Trait` is not implemented
    fn uhoh() {}
}

struct Baz {
//~^ ERROR not implemented
    a: Foo<int>,
}

enum Boo {
//~^ ERROR not implemented
    Quux(Bar<uint>),
}

struct Badness<T> {
//~^ ERROR not implemented
    b: Foo<T>,
}

enum MoreBadness<T> {
//~^ ERROR not implemented
    EvenMoreBadness(Bar<T>),
}

trait PolyTrait<T> {
    fn whatever() {}
}

struct Struct;

impl PolyTrait<Foo<uint>> for Struct {
//~^ ERROR not implemented
    fn whatever() {}
}

fn main() {
}

