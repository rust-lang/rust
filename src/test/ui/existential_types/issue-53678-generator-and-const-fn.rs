// check-pass

#![feature(const_fn, generators, generator_trait, existential_type)]

use std::ops::Generator;

existential type GenOnce<Y, R>: Generator<Yield = Y, Return = R>;

const fn const_generator<Y, R>(yielding: Y, returning: R) -> GenOnce<Y, R> {
    move || {
        yield yielding;

        return returning;
    }
}

const FOO: GenOnce<usize, usize> = const_generator(10, 100);

fn main() {}
