#![feature(generators, generator_trait, rustc_attrs)]
#![feature(type_alias_impl_trait)]

// check-pass

mod gen {
    use std::ops::Generator;

    pub type GenOnce<Y, R> = impl Generator<Yield = Y, Return = R>;

    pub const fn const_generator<Y, R>(yielding: Y, returning: R) -> GenOnce<Y, R> {
        move || {
            yield yielding;

            return returning;
        }
    }
}

const FOO: gen::GenOnce<usize, usize> = gen::const_generator(10, 100);

fn main() {}
