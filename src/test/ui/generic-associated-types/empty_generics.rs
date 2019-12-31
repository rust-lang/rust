#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait Foo {
    type Bar<,>;
    //~^ ERROR expected one of `>`, `const`, identifier, or lifetime, found `,`
}

fn main() {}
