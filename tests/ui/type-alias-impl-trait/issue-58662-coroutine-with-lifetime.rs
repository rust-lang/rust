//@ check-pass

#![feature(coroutines, coroutine_trait)]
#![feature(type_alias_impl_trait)]

use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

type RandCoroutine<'a> = impl Coroutine<Return = (), Yield = u64> + 'a;
#[define_opaque(RandCoroutine)]
fn rand_coroutine<'a>(rng: &'a ()) -> RandCoroutine<'a> {
    #[coroutine]
    move || {
        let _rng = rng;
        loop {
            yield 0;
        }
    }
}

pub type RandCoroutineWithIndirection<'c> = impl Coroutine<Return = (), Yield = u64> + 'c;
#[define_opaque(RandCoroutineWithIndirection)]
pub fn rand_coroutine_with_indirection<'a>(rng: &'a ()) -> RandCoroutineWithIndirection<'a> {
    fn helper<'b>(rng: &'b ()) -> impl 'b + Coroutine<Return = (), Yield = u64> {
        #[coroutine]
        move || {
            let _rng = rng;
            loop {
                yield 0;
            }
        }
    }

    helper(rng)
}

fn main() {
    let mut gen = rand_coroutine(&());
    match unsafe { Pin::new_unchecked(&mut gen) }.resume(()) {
        CoroutineState::Yielded(_) => {}
        CoroutineState::Complete(_) => {}
    };
}
