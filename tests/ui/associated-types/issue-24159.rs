//@ check-pass

#![allow(unused)]

trait Bar<T> {
    fn dummy(&self);
}

trait Foo {
    type A;
    type B: Bar<Self::A>;

    fn get_b(&self) -> &Self::B;
}

fn test_bar<A, B: Bar<A>>(_: &B) {}

fn test<A, F: Foo<A = A>>(f: &F) {
    test_bar(f.get_b());
}

trait Bar1<T> {}
trait Caz1 {
    type A;
    type B: Bar1<Self::A>;
}

fn test1<T, U>() where T: Caz1, U: Caz1<A = T::A> {}

trait Bar2<T> {}
trait Caz2 {
    type A;
    type B: Bar2<Self::A>;
}
fn test2<T: Caz2<A = ()>>() {}

fn main() {}
