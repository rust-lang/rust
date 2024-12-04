//@ check-pass

trait Foo: Bar<Out = ()> {}
trait Bar {
    type Out;
}

fn w(x: &dyn Foo<Out = ()>) {
    //~^ WARN associated type bound for `Out` in `dyn Foo` is redundant
    let x: &dyn Foo = x;
}

fn main() {}
