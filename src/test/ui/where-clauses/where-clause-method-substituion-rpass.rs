// run-pass
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

trait Foo<T> { fn dummy(&self, arg: T) { } }

trait Bar<A> {
    fn method<B>(&self) where A: Foo<B>;
}

struct S;
struct X;

impl Foo<S> for X {}

impl Bar<X> for i32 {
    fn method<U>(&self) where X: Foo<U> {
    }
}

fn main() {
    1.method::<S>();
}
