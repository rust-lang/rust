#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;

fn crash(x: Foo) -> Foo {
    //~^ ERROR: overflow
    x
}

fn main() {}
