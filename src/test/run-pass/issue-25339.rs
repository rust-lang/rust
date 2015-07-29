// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_type_defaults)]

use std::marker::PhantomData;

pub trait Routing<I> {
    type Output;
    fn resolve(&self, input: I);
}

pub trait ToRouting {
    type Input;
    type Routing : ?Sized = Routing<Self::Input, Output=()>;
    fn to_routing(self) -> Self::Routing;
}

pub struct Mount<I, R: Routing<I>> {
    action: R,
    _marker: PhantomData<I>
}

impl<I, R: Routing<I>> Mount<I, R> {
    pub fn create<T: ToRouting<Routing=R>>(mount: &str, input: T) {
        input.to_routing();
    }
}

fn main() {
}
