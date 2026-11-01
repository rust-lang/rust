//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn firstn() -> impl Coroutine<Yield = u64, Return = ()> {
    #[coroutine]
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
    let mut coroutine_iterator_2 = {
        let mut coroutine_iterator = Box::new(CoroutineIteratorAdapter(firstn()));
        coroutine_iterator.next(); // pin it

        Box::new(*coroutine_iterator) // move it
    }; // *deallocate* coroutine_iterator

    coroutine_iterator_2.next(); // and use moved value
}
