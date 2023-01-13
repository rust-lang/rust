#![feature(generators, generator_trait, never_type)]

use std::ops::Generator;

fn mk_gen() -> impl Generator<Return=!, Yield=()> {
    || { loop { yield; } }
}

fn main() {
    let gens: [impl Generator<Return=!, Yield=()>;2] = [ mk_gen(), mk_gen() ];
    //~^ `impl Trait` only allowed in function and inherent method return types
}
