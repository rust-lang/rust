//@compile-flags: --edition 2024 -Zunstable-options
#![feature(coroutines, coroutine_trait, gen_blocks)]

use std::ops::Coroutine;

fn foo() -> impl Coroutine<Yield = u32, Return = ()> { //~ ERROR the trait `Coroutine` is not implemented for
    gen { yield 42 }
}

fn bar() -> impl Coroutine<Yield = i64, Return = ()> { //~ ERROR the trait `Coroutine` is not implemented for
    gen { yield 42 }
}

fn baz() -> impl Coroutine<Yield = i32, Return = ()> { //~ ERROR the trait `Coroutine` is not implemented for
    gen { yield 42 }
}

fn main() {}
