trait NotFoo {}

pub trait Foo: NotFoo {
    type OnlyFoo;
}

pub trait Service {
    type AssocType;
}

pub trait ThriftService<Bug: NotFoo>:
//~^ ERROR the trait bound `Bug: Foo` is not satisfied
//~| ERROR the trait bound `Bug: Foo` is not satisfied
    Service<AssocType = <Bug as Foo>::OnlyFoo>
{
    fn get_service(
    //~^ ERROR the trait bound `Bug: Foo` is not satisfied
    //~| ERROR the trait bound `Bug: Foo` is not satisfied
        &self,
    ) -> Self::AssocType;
}

fn with_factory<H>(factory: dyn ThriftService<()>) {}
//~^ ERROR the trait bound `(): Foo` is not satisfied

fn main() {}
