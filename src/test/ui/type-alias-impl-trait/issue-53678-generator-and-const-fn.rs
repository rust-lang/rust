#![feature(generators, generator_trait, rustc_attrs)]
#![feature(type_alias_impl_trait)]

use std::ops::Generator;

type GenOnce<Y, R> = impl Generator<Yield = Y, Return = R>;

const fn const_generator<Y, R>(yielding: Y, returning: R) -> GenOnce<Y, R> {
    move || {
        yield yielding;

        return returning;
    }
}

const FOO: GenOnce<usize, usize> = const_generator(10, 100);

#[rustc_error]
fn main() {} //~ ERROR
