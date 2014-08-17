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

fn explode(x: Foo<uint>) {}
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type

fn kaboom(y: Bar<f32>) {}
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type

impl<T> Foo<T> {
    fn uhoh() {}
}

struct Baz {
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type
    a: Foo<int>,
}

enum Boo {
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type
    Quux(Bar<uint>),
}

struct Badness<T> {
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type
    b: Foo<T>,
}

enum MoreBadness<T> {
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type
    EvenMoreBadness(Bar<T>),
}

trait PolyTrait<T> {
    fn whatever() {}
}

struct Struct;

impl PolyTrait<Foo<uint>> for Struct {
//~^ ERROR failed to find an implementation
//~^^ ERROR instantiating a type parameter with an incompatible type
    fn whatever() {}
}

fn main() {
    let bar: Bar<f64> = return;
    //~^ ERROR failed to find an implementation
    //~^^ ERROR instantiating a type parameter with an incompatible type
    let _ = bar;
}

