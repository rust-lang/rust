//@ check-pass

trait Trait1 {
    type Assoc1: Bar;

    fn assoc(self) -> Self::Assoc1;
}

impl Trait1 for () {
    type Assoc1 = ();
    fn assoc(self) {}
}

trait Foo {}
impl Foo for () {}
trait Bar {}
impl Bar for () {}

fn hello() -> impl Trait1<Assoc1: Foo> {
    ()
}

fn world() {
    // Tests that `Assoc1: Foo` bound in the RPIT doesn't disqualify
    // the `Assoc1: Bar` bound in the item, as a nested RPIT desugaring
    // would do.

    fn is_foo(_: impl Foo) {}
    is_foo(hello().assoc());

    fn is_bar(_: impl Bar) {}
    is_bar(hello().assoc());
}

fn main() {}
