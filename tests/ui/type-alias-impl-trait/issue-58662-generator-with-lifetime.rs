// check-pass

#![feature(generators, generator_trait)]
#![feature(type_alias_impl_trait)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

type RandGenerator<'a> = impl Generator<Return = (), Yield = u64> + 'a;
fn rand_generator<'a>(rng: &'a ()) -> RandGenerator<'a> {
    move || {
        let _rng = rng;
        loop {
            yield 0;
        }
    }
}

pub type RandGeneratorWithIndirection<'c> = impl Generator<Return = (), Yield = u64> + 'c;
pub fn rand_generator_with_indirection<'a>(rng: &'a ()) -> RandGeneratorWithIndirection<'a> {
    fn helper<'b>(rng: &'b ()) -> impl 'b + Generator<Return = (), Yield = u64> {
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
    let mut gen = rand_generator(&());
    match unsafe { Pin::new_unchecked(&mut gen) }.resume(()) {
        GeneratorState::Yielded(_) => {}
        GeneratorState::Complete(_) => {}
    };
}
