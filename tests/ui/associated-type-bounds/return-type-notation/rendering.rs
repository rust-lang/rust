//@ run-rustfix

#![allow(unused, todo_macro_uses)]
#![feature(return_type_notation)]

trait Foo {
    fn missing() -> impl Sized;
}

impl Foo for () {
    //~^ ERROR not all trait items implemented, missing: `missing`
}

fn main() {}
