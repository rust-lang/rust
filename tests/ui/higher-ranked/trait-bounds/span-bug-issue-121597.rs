#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

trait Foo: for<T> Bar<T> {}

trait Bar<T: ?Sized> {
    fn method(&self) {}
}

struct Type2;
fn needs_bar(_: *mut Type2) {}

fn main() {
    let x: &dyn Foo = &();
    //~^ ERROR the trait `Foo` is not dyn compatible

    needs_bar(x);
    //~^ ERROR mismatched types
}
