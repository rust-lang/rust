trait Foo {}
trait Bar {}

impl<T> Foo for T where T: Bar {}
fn needs_foo(_: impl Foo) {}

trait Mirror {
    type Mirror;
}
impl<T> Mirror for T {
    type Mirror = T;
}

// Make sure the `Alias: Foo` bound doesn't "shadow" the impl, since the
// impl is really the only candidate we care about here for the purpose
// of error reporting.
fn hello<T>() where <T as Mirror>::Mirror: Foo {
    needs_foo(());
    //~^ ERROR the trait bound `(): Foo` is not satisfied
}

fn main() {}
