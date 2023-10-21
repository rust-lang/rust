#![feature(coroutines, coroutine_trait, never_type)]

use std::ops::Coroutine;

fn mk_gen() -> impl Coroutine<Return=!, Yield=()> {
    || { loop { yield; } }
}

fn main() {
    let gens: [impl Coroutine<Return=!, Yield=()>;2] = [ mk_gen(), mk_gen() ];
    //~^ `impl Trait` only allowed in function and inherent method argument and return types
}
