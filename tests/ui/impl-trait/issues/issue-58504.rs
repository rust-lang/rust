#![feature(coroutines, coroutine_trait, never_type)]

use std::ops::Coroutine;

fn mk_gen() -> impl Coroutine<Return=!, Yield=()> {
    #[coroutine] || { loop { yield; } }
}

fn main() {
    let gens: [impl Coroutine<Return=!, Yield=()>;2] = [ mk_gen(), mk_gen() ];
    //~^ ERROR `impl Trait` is not allowed in the type of variable bindings
}
