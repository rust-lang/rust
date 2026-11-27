//@ check-pass

trait Foo: Bar<Out = ()> {}
trait Bar {
    type Out;
}

fn w(x: &dyn Foo<Out = ()>) {
    let x: &dyn Foo = x;
}

fn main() {}
