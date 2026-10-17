//@ edition: 2024
//@ compile-flags: --diagnostic-width=300
#![feature(coroutines, coroutine_trait, gen_blocks)]

use std::ops::Coroutine;

fn foo() -> impl for<'y> Coroutine<Yield<'y> = u32, Return = ()> { //~ ERROR: Coroutine` is not satisfied
    gen { yield 42 }
}

fn bar() -> impl for<'y> Coroutine<Yield<'y> = i64, Return = ()> { //~ ERROR: Coroutine` is not satisfied
    gen { yield 42 }
}

fn baz() -> impl for<'y> Coroutine<Yield<'y> = i32, Return = ()> { //~ ERROR: Coroutine` is not satisfied
    gen { yield 42 }
}

fn main() {}
