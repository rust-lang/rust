#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

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

// FIXME(#86731): The below is illegal use of `type_alias_impl_trait`
// but the compiler doesn't report it, we should fix it.
pub type FooImpl = impl Foo;
pub type BarImpl = impl Bar<Foo = FooImpl>;
//~^ ERROR: type mismatch resolving `<() as Bar>::Foo == ()`

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
