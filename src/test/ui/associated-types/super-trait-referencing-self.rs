// check-pass
trait Foo {
    type Bar;
}
trait Qux: Foo + AsRef<Self::Bar> {}
trait Foo2 {}

trait Qux2: Foo2 + AsRef<Self::Bar> {
    type Bar;
}

fn main() {}
