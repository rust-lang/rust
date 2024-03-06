#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;

fn foo() -> Foo {
    //~^ ERROR: overflow
    foo
}

fn main() {}
