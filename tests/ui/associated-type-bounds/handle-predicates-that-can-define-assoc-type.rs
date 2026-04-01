//@ check-pass

trait Foo<T> {}
trait Bar {
    type A;
    type B;
}
trait Baz: Bar<B = u32> + Foo<Self::A> {}

fn main() {}
