// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#![feature(default_type_parameter_fallback)]

use std::marker::PhantomData;

#[derive(Copy, Clone)]
enum Opt<T> {
    Som(T),
    Non,
}

trait Id {
    type Me;
}

impl<A> Id for A {
    type Me = A;
}

struct Foo<X, Y, Z> {
    data: PhantomData<(X, Y, Z)>,
}

impl<X: Default = u32, Y = <X as Id>::Me, Z = <Y as Id>::Me>
    Foo<X, Y, Z> {
    fn new(_: Opt<X>, _: Opt<Y>, _: Opt<Z>) -> Foo<X, Y, Z> {
        Foo { data: PhantomData }
    }
}

fn main() {
    let a = Opt::Non;
    let _ = Foo::new(a, a, a);
}
