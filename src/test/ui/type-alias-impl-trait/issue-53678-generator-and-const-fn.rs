#![feature(const_impl_trait, generators, generator_trait, rustc_attrs)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(impl_trait_in_bindings, type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete
//[full_tait]~| WARN incomplete

use std::ops::Generator;

type GenOnce<Y, R> = impl Generator<Yield = Y, Return = R>;

const fn const_generator<Y, R>(yielding: Y, returning: R) -> GenOnce<Y, R> {
    move || {
        yield yielding;

        return returning;
    }
}

const FOO: GenOnce<usize, usize> = const_generator(10, 100); //[min_tait]~ ERROR not permitted here

#[rustc_error]
fn main() {} //[full_tait]~ ERROR
