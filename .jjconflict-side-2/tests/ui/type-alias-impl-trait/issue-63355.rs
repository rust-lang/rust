#![feature(type_alias_impl_trait)]

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

    #[define_opaque(FooImpl)]
    fn foo() -> Self::Foo {
        ()
    }
}

pub type FooImpl = impl Foo;
pub type BarImpl = impl Bar<Foo = FooImpl>;

impl Baz for () {
    type Foo = FooImpl;
    type Bar = BarImpl;

    #[define_opaque(FooImpl)]
    fn foo() -> Self::Foo {
        ()
    }

    #[define_opaque(BarImpl)]
    fn bar() -> Self::Bar {
        //~^ ERROR: item does not constrain `FooImpl::{opaque#0}`
        ()
    }
}

fn main() {}
