trait Foo<'s> {}

impl<'s> Foo<'s> for () {}

struct Bar;

impl<'s, T: Foo<'s>> From<T> for Bar {
    fn from(_: T) -> Self {
        Bar
    }
}

fn main() {
    let _: Bar = ((),).into();
    //~^ ERROR he trait bound `((),): Into<Bar>` is not satisfied
}
