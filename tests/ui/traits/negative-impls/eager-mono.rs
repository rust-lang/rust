//@ build-pass
//@ compile-flags:-C link-dead-code=y

#![feature(negative_impls)]

trait Foo {
    fn foo() {}
}

impl !Foo for () {}

fn main() {}
