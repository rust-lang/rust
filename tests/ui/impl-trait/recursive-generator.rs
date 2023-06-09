#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};

fn foo() -> impl Generator<Yield = (), Return = ()> {
    //~^ ERROR cannot resolve opaque type
    //~| NOTE recursive opaque type
    //~| NOTE in this expansion of desugaring of
    || {
    //~^ NOTE returning here
        let mut gen = Box::pin(foo());
        //~^ NOTE generator captures itself here
        let mut r = gen.as_mut().resume(());
        while let GeneratorState::Yielded(v) = r {
            yield v;
            r = gen.as_mut().resume(());
        }
    }
}

fn main() {
    foo();
}
