trait Bar {}

trait Foo {
    type Assoc: Bar;
}

impl Foo for () {
    // Doesn't error because we abort compilation after the errors below.
    // See point-at-type-on-obligation-failure-3.rs
    type Assoc = bool;
}

trait Baz where Self::Assoc: Bar {
    type Assoc;
}

impl Baz for () {
    type Assoc = bool; //~ ERROR the trait bound `bool: Bar` is not satisfied
}

trait Bat where <Self as Bat>::Assoc: Bar {
    type Assoc;
}

impl Bat for () {
    type Assoc = bool; //~ ERROR the trait bound `bool: Bar` is not satisfied
}

fn main() {}
