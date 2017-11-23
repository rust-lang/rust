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

trait Id {
    type This;
}

impl<A> Id for A {
    type This = A;
}

struct Foo<X: Default = usize, Y = <X as Id>::This> {
    data: PhantomData<(X, Y)>
}

impl<X: Default, Y> Foo<X, Y> {
    fn new() -> Foo<X, Y> {
        Foo { data: PhantomData }
    }
}

fn main() {
    let foo = Foo::new();
}
