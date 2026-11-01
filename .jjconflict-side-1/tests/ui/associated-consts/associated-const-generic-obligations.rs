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
    //~^ ERROR implemented const `FROM` has an incompatible type for trait [E0326]
}

fn main() {}
