#![feature(type_alias_impl_trait)]

// check-pass

type Foo = impl Fn() -> Foo;

fn foo() -> Foo {
    foo
}

fn main() {}
