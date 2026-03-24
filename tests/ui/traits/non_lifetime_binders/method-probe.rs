//@ check-pass

#![feature(non_lifetime_binders)]

trait Foo: for<T> Bar<T> {}

trait Bar<T> {
    fn method() -> T;
}

fn x<T: Foo>() {
    let _: i32 = T::method();
}

fn main() {}
