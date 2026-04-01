//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![feature(associated_type_defaults)]

use std::marker::PhantomData;

pub trait Routing<I> {
    type Output;
    fn resolve(&self, input: I);
}

pub trait ToRouting {
    type Input;
    type Routing : ?Sized = dyn Routing<Self::Input, Output=()>;
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
