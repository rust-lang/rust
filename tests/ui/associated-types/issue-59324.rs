trait NotFoo {}

pub trait Foo: NotFoo {
    type OnlyFoo;
}

pub trait Service {
    type AssocType;
}

pub trait ThriftService<Bug: NotFoo>:
//~^ ERROR trait `Foo` is not implemented for `Bug`
//~| ERROR trait `Foo` is not implemented for `Bug`
    Service<AssocType = <Bug as Foo>::OnlyFoo>
{
    fn get_service(
    //~^ ERROR trait `Foo` is not implemented for `Bug`
        &self,
    ) -> Self::AssocType;
    //~^ ERROR trait `Foo` is not implemented for `Bug`
}

fn with_factory<H>(factory: dyn ThriftService<()>) {}
//~^ ERROR trait `Foo` is not implemented for `()`
//~| ERROR trait `Foo` is not implemented for `()`
//~| ERROR cannot be known at compilation time

fn main() {}
