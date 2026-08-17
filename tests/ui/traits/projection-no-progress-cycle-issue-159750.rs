// Regression test for #159750. When `<T as Trait>::Assoc` can't be resolved
// to a concrete type, equating it with an inference variable spawns a copy of
// the obligation being processed, which fulfillment used to treat as progress
// and loops forever. Check that we report an error instead of hanging.

trait Container<T> {
    type IDContainer;
}

impl<T> Container<T> for Option<T> {
    type IDContainer = ();
}

trait Queryable {
    type Output;
    type Container;

    fn finish(&self) -> <Self::Container as Container<Self::Output>>::IDContainer;
    //~^ ERROR the trait bound `<Self as Queryable>::Container: Container<<Self as Queryable>::Output>` is not satisfied
}

fn follow<E>() -> impl Queryable<Container = Option<E>> {
    //~^ ERROR the trait bound `(): Queryable` is not satisfied
    loop {}
}

fn main() {
    follow().finish();
    //~^ ERROR type annotations needed
}
