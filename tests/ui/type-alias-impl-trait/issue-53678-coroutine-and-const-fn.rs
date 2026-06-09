#![feature(coroutines, coroutine_trait, rustc_attrs, const_async_blocks)]
#![feature(type_alias_impl_trait)]

//@ check-pass

use std::ops::Coroutine;

pub type CoroOnce<Y, R> = impl Coroutine<Yield = Y, Return = R>;

#[define_opaque(CoroOnce)]
pub const fn const_coroutine<Y, R>(yielding: Y, returning: R) -> CoroOnce<Y, R> {
    #[coroutine]
    move || {
        yield yielding;

        return returning;
    }
}

const FOO: CoroOnce<usize, usize> = const_coroutine(10, 100);

fn main() {}
