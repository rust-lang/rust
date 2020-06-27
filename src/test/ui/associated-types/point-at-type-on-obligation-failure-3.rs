trait Bar {}

trait Foo {
    type Assoc: Bar;
}

impl Foo for () {
    type Assoc = bool; //~ ERROR the trait bound `bool: Bar` is not satisfied
}

fn main() {}
