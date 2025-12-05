#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {}

trait Trait {
    reuse to_reuse::foo { foo }
    //~^ ERROR cannot find function `foo` in module `to_reuse`
    //~| ERROR cannot find value `foo` in this scope
}

fn main() {}
