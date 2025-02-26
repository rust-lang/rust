trait Foo {
    fn foo<S>(s: S) -> impl Sized;
}

trait Bar {}

impl Foo for () {
    fn foo<S>(s: S) -> impl Sized where S: Bar {}
    //~^ ERROR impl has stricter requirements than trait
}

fn main() {}
