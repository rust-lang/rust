//@ edition: 2024
//@ compile-flags: --diagnostic-width=300
#![feature(coroutines, coroutine_trait, gen_blocks)]

use std::ops::Coroutine;

fn foo() -> impl Coroutine<Yield = u32, Return = ()> { //~ ERROR: Coroutine` is not satisfied
    gen { 42.yield }
}

fn bar() -> impl Coroutine<Yield = i64, Return = ()> { //~ ERROR: Coroutine` is not satisfied
    gen { 42.yield }
}

fn baz() -> impl Coroutine<Yield = i32, Return = ()> { //~ ERROR: Coroutine` is not satisfied
    gen { 42.yield }
}

fn main() {}
