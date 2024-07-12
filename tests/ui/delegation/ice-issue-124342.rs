#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {}

trait Trait {
    reuse to_reuse::foo { foo }
    //~^ ERROR cannot find function `foo`
    //~| ERROR cannot find value `foo`
}

fn main() {}
