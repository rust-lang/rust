trait NotFoo {}

pub trait Foo: NotFoo {
    type OnlyFoo;
}

pub trait Service {
    type AssocType;
}

pub trait ThriftService<Bug: NotFoo>:
//~^ ERROR the trait bound `Bug: Foo` is not satisfied
    Service<AssocType = <Bug as Foo>::OnlyFoo>
//~^ ERROR the trait bound `Bug: Foo` is not satisfied
{
    fn get_service(
    //~^ ERROR the trait bound `Bug: Foo` is not satisfied
    //~| ERROR the trait bound `Bug: Foo` is not satisfied
        &self,
    ) -> Self::AssocType;
    //~^ ERROR the trait bound `Bug: Foo` is not satisfied
}

fn with_factory<H>(factory: dyn ThriftService<()>) {}
//~^ ERROR the trait bound `(): Foo` is not satisfied
//~| ERROR the trait bound `(): Foo` is not satisfied
//~| ERROR cannot be known at compilation time

fn main() {}
