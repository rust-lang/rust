//@ build-fail

trait A {
    type Assoc: ?Sized;
}

impl A for () {
    type Assoc = Foo<()>;
}
struct Foo<T: A>(T::Assoc);
//~^ ERROR cycle detected when computing layout of `Foo<()>`

fn main() {
    let x: Foo<()>;
}
