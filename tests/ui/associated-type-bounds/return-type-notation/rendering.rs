//@ run-rustfix

#![allow(unused)]
#![feature(return_type_notation)]

trait Foo {
    fn missing() -> impl Sized;
}

impl Foo for () {
    //~^ ERROR not all trait items implemented, missing: `missing`
}

fn main() {}
