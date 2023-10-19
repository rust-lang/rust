//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows
#![feature(generators, generator_trait)]

use std::{
    ops::{Coroutine, CoroutineState},
    pin::Pin,
};

fn firstn() -> impl Coroutine<Yield = u64, Return = ()> {
    static move || {
        let mut num = 0;
        let num = &mut num;
        *num += 0;

        yield *num;
        *num += 1; //~ERROR: has been freed
    }
}

struct CoroutineIteratorAdapter<G>(G);

impl<G> Iterator for CoroutineIteratorAdapter<G>
where
    G: Coroutine<Return = ()>,
{
    type Item = G::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        let me = unsafe { Pin::new_unchecked(&mut self.0) };
        match me.resume(()) {
            CoroutineState::Yielded(x) => Some(x),
            CoroutineState::Complete(_) => None,
        }
    }
}

fn main() {
    let mut generator_iterator_2 = {
        let mut generator_iterator = Box::new(CoroutineIteratorAdapter(firstn()));
        generator_iterator.next(); // pin it

        Box::new(*generator_iterator) // move it
    }; // *deallocate* generator_iterator

    generator_iterator_2.next(); // and use moved value
}
