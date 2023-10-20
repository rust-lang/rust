#![feature(coroutines, coroutine_trait, rustc_attrs)]
#![feature(type_alias_impl_trait)]

// check-pass

mod gen {
    use std::ops::Coroutine;

    pub type CoroOnce<Y, R> = impl Coroutine<Yield = Y, Return = R>;

    pub const fn const_coroutine<Y, R>(yielding: Y, returning: R) -> CoroOnce<Y, R> {
        move || {
            yield yielding;

            return returning;
        }
    }
}

const FOO: gen::CoroOnce<usize, usize> = gen::const_coroutine(10, 100);

fn main() {}
