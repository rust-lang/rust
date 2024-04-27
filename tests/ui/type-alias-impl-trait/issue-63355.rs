#![feature(type_alias_impl_trait)]
//@ check-pass

pub trait Foo {}

pub trait Bar {
    type Foo: Foo;

    fn foo() -> Self::Foo;
}

pub trait Baz {
    type Foo: Foo;
    type Bar: Bar<Foo = Self::Foo>;

    fn foo() -> Self::Foo;
    fn bar() -> Self::Bar;
}

impl Foo for () {}

impl Bar for () {
    type Foo = FooImpl;

    fn foo() -> Self::Foo {
        ()
    }
}

pub type FooImpl = impl Foo;
pub type BarImpl = impl Bar<Foo = FooImpl>;

impl Baz for () {
    type Foo = FooImpl;
    type Bar = BarImpl;

    fn foo() -> Self::Foo {
        ()
    }

    fn bar() -> Self::Bar {
        ()
    }
}

fn main() {}
