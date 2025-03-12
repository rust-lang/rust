//@ run-rustfix

#![allow(unused)]

trait Foo {
    fn missing() -> impl Sized;
}

impl Foo for () {
    //~^ ERROR not all trait items implemented, missing: `missing`
}

fn main() {}
