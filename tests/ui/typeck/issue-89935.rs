//@ check-pass

trait Foo: Baz {}
trait Bar {}
trait Baz: Bar {
    fn bar(&self);
}

impl<T: Foo> Bar for T {}
impl<T: Foo> Baz for T {
    fn bar(&self) {}
}

fn accept_foo(x: Box<dyn Foo>) {
    x.bar();
}

fn main() {}
