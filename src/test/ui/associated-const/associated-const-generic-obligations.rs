trait Foo {
    type Out: Sized;
}

impl Foo for String {
    type Out = String;
}

trait Bar: Foo {
    const FROM: Self::Out;
}

impl<T: Foo> Bar for T {
    const FROM: &'static str = "foo";
    //~^ ERROR the trait bound `T: Foo` is not satisfied [E0277]
}

fn main() {}
