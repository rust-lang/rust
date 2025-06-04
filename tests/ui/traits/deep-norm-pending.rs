trait Foo {
    type Assoc;
}

trait Bar {
    fn method() -> impl Sized;
    //~^ ERROR the trait bound `T: Foo` is not satisfied
}
impl<T> Bar for T
//~^ ERROR the trait bound `T: Foo` is not satisfied
where
    <T as Foo>::Assoc: Sized,
    //~^ ERROR the trait bound `T: Foo` is not satisfied
    //~| ERROR the trait bound `T: Foo` is not satisfied
{
    fn method() {}
    //~^ ERROR the trait bound `T: Foo` is not satisfied
    //~| ERROR the trait bound `T: Foo` is not satisfied
    //~| ERROR the trait bound `T: Foo` is not satisfied
    //~| ERROR the trait bound `T: Foo` is not satisfied
    //~| ERROR the trait bound `T: Foo` is not satisfied
}

fn main() {}
